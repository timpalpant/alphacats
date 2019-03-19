// Package model implements an LSTM-based network model for use in DeepCFR.
// As input, the model takes the public game history (one-hot encoded),
// as well as the player's current hand (one-hot encoded) and predicts the
// advantages for each possible action in this infoset.
package model

import (
	"bytes"
	"encoding/gob"
	"expvar"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/golang/glog"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats/gamestate"
)

const (
	graphTag               = "lstm"
	outputLayer            = "output/mul"
	maxPredictionBatchSize = 4096
)

var (
	samplesPredicted = expvar.NewInt("num_predicted_samples")
	batchesPredicted = expvar.NewInt("num_predicted_batches")
)

// tfConfig is tf.ConfigProto(
//   gpu_options=tf.GPUOptions(allow_growth=True)
// ).SerializeToString()
var tfConfig = []byte{50, 2, 32, 1}

type Params struct {
	BatchSize      int
	ModelOutputDir string
}

// LSTM is a model for AlphaCats to be used with DeepCFR
// and implements deepcfr.Model.
type LSTM struct {
	params Params
	iter   int
}

func NewLSTM(p Params) *LSTM {
	return &LSTM{params: p}
}

// Train implements deepcfr.Model.
func (m *LSTM) Train(samples deepcfr.Buffer) deepcfr.TrainedModel {
	glog.Infof("Training network with %d samples", len(samples.GetSamples()))
	// Save training data to disk in a tempdir.
	tmpDir, err := ioutil.TempDir("", "alphacats-training-data-")
	if err != nil {
		glog.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	glog.Infof("Saving training data to: %v", tmpDir)
	trainingData := samples.GetSamples()
	if err := saveTrainingData(trainingData, tmpDir, m.params.BatchSize); err != nil {
		glog.Fatal(err)
	}

	// Shell out to Python to train the network.
	outputDir := filepath.Join(m.params.ModelOutputDir, fmt.Sprintf("model_%08d", m.iter))
	cmd := exec.Command("python", "model/train.py", tmpDir, outputDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	glog.Infof("Running: %v", cmd.Args)
	start := time.Now()
	if err := cmd.Run(); err != nil {
		glog.Fatal(err)
	}

	elapsed := time.Since(start)
	sps := float64(len(trainingData)) / elapsed.Seconds()
	glog.V(1).Infof("Finished training (took %v, %v samples/sec)",
		elapsed, sps)
	m.iter++

	// Load trained model.
	trained, err := LoadTrainedLSTM(outputDir)
	if err != nil {
		glog.Fatal(err)
	}

	return trained
}

func (m *LSTM) GobDecode(buf []byte) error {
	r := bytes.NewReader(buf)
	dec := gob.NewDecoder(r)

	if err := dec.Decode(&m.params); err != nil {
		return err
	}

	if err := dec.Decode(&m.iter); err != nil {
		return err
	}

	return nil
}

func (m *LSTM) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(m.params); err != nil {
		return nil, err
	}

	if err := enc.Encode(m.iter); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

var reqPool = sync.Pool{
	New: func() interface{} {
		return &predictionRequest{
			resultCh: make(chan []float32, 1),
		}
	},
}

type TrainedLSTM struct {
	dir   string
	model *tf.SavedModel
	reqCh chan *predictionRequest
}

func LoadTrainedLSTM(dir string) (*TrainedLSTM, error) {
	opts := &tf.SessionOptions{Config: tfConfig}
	model, err := tf.LoadSavedModel(dir, []string{graphTag}, opts)
	if err != nil {
		return nil, err
	}

	m := &TrainedLSTM{
		dir:   dir,
		model: model,
		reqCh: make(chan *predictionRequest, 1),
	}
	go m.bgPredictionHandler()
	return m, nil
}

// Predict implements deepcfr.TrainedModel.
func (m *TrainedLSTM) Predict(infoSet cfr.InfoSet, nActions int) []float32 {
	is := infoSet.(*gamestate.InfoSet)
	req := reqPool.Get().(*predictionRequest)
	req.history = EncodeHistory(is.History)
	req.hand = encodeHand(is.Hand)
	req.nActions = maskNumActions(nActions)

	m.reqCh <- req
	prediction := <-req.resultCh
	reqPool.Put(req)

	glog.V(3).Infof("Predicted advantages: %v", prediction)
	advantages := prediction[:nActions]
	makePositive(advantages)
	total := sum(advantages)

	if total > 0 {
		for i := range advantages {
			advantages[i] /= total
		}
	} else { // Uniform probability.
		for i := range advantages {
			advantages[i] = 1.0 / float32(len(advantages))
		}
	}

	return advantages
}

// When serializing, we just serialize the path to the model already on disk.
func (m *TrainedLSTM) GobDecode(buf []byte) error {
	dir := string(buf)
	model, err := LoadTrainedLSTM(dir)
	*m = *model
	return err
}

func (m *TrainedLSTM) GobEncode() ([]byte, error) {
	return []byte(m.dir), nil
}

func (m *TrainedLSTM) Close() {
	close(m.reqCh)
	// TODO: Wait for final batch, if there is one.
	m.model.Session.Close()
}

type predictionRequest struct {
	history  [][]float32
	hand     []float32
	nActions []float32
	resultCh chan []float32
}

// Handles prediction requests, attempting to batch all pending requests.
func (m *TrainedLSTM) bgPredictionHandler() {
	// No buffer here because we are collecting batches and we want the batch to be as
	// large as possible until we are ready to process it.
	encodeCh := make(chan []*predictionRequest)
	defer close(encodeCh)
	go handleEncoding(m.model, encodeCh)

	for {
		// Wait for first request so we know we have at least one.
		req, ok := <-m.reqCh
		if !ok {
			return
		}

		// Drain any additional requests as long as we're waiting for an encoding spot.
		batch := []*predictionRequest{req}
	Loop:
		for {
			select {
			case req, ok := <-m.reqCh:
				if !ok {
					encodeCh <- batch
					return
				}

				batch = append(batch, req)
				if len(batch) >= maxPredictionBatchSize {
					encodeCh <- batch
					break Loop
				}
			case encodeCh <- batch:
				break Loop
			}
		}
	}
}

func handleEncoding(model *tf.SavedModel, batchCh chan []*predictionRequest) {
	outputCh := make(chan *batchPredictionRequest, 1)
	defer close(outputCh)
	go handleBatchPredictions(model, outputCh)

	for batch := range batchCh {
		histories := make([][][]float32, len(batch))
		hands := make([][]float32, len(batch))
		nActions := make([][]float32, len(batch))
		for i, req := range batch {
			histories[i] = req.history
			hands[i] = req.hand
			nActions[i] = req.nActions
		}

		historyTensor, err := tf.NewTensor(histories)
		if err != nil {
			glog.Fatal(err)
		}

		handTensor, err := tf.NewTensor(hands)
		if err != nil {
			glog.Fatal(err)
		}

		nActionsTensor, err := tf.NewTensor(nActions)
		if err != nil {
			glog.Fatal(err)
		}

		outputCh <- &batchPredictionRequest{
			history:  historyTensor,
			hand:     handTensor,
			nActions: nActionsTensor,
			batch:    batch,
		}
	}
}

type batchPredictionRequest struct {
	history  *tf.Tensor
	hand     *tf.Tensor
	nActions *tf.Tensor
	batch    []*predictionRequest
}

func handleBatchPredictions(model *tf.SavedModel, reqCh chan *batchPredictionRequest) {
	for req := range reqCh {
		result := predictBatch(model, req.history, req.hand, req.nActions)
		for i, req := range req.batch {
			req.resultCh <- result[i]
		}

		samplesPredicted.Add(int64(len(req.batch)))
		batchesPredicted.Add(1)
	}
}

func predictBatch(model *tf.SavedModel, history, hand, nActions *tf.Tensor) [][]float32 {
	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("history").Output(0):     history,
			model.Graph.Operation("hand").Output(0):        hand,
			model.Graph.Operation("num_actions").Output(0): nActions,
		},
		[]tf.Output{
			model.Graph.Operation(outputLayer).Output(0),
		},
		nil,
	)

	if err != nil {
		glog.Fatal(err)
	}

	return result[0].Value().([][]float32)
}

func makePositive(v []float32) {
	for i := range v {
		if v[i] < 0 {
			v[i] = 0.0
		}
	}
}

func sum(v []float32) float32 {
	var total float32
	for _, x := range v {
		total += x
	}

	return total
}

func init() {
	gob.Register(&LSTM{})
	gob.Register(&TrainedLSTM{})
}
