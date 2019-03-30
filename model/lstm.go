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
	"time"

	"github.com/golang/glog"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats"
)

const (
	graphTag               = "lstm"
	outputLayer            = "output/BiasAdd"
	maxPredictionBatchSize = 4096
	numEncodingWorkers     = 4
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
	is := infoSet.(*alphacats.InfoSetWithAvailableActions)
	if len(is.AvailableActions) != nActions {
		panic(fmt.Errorf("InfoSet has %d actions but expected %d",
			len(is.AvailableActions), nActions))
	}

	history := EncodeHistory(is.History)
	hand := encodeHand(is.Hand)
	reqs := make([]*predictionRequest, nActions)
	for i, action := range is.AvailableActions {
		req := &predictionRequest{
			history:  history,
			hand:     hand,
			action:   encodeAction(action),
			resultCh: make(chan float32, 1),
		}

		m.reqCh <- req
		reqs[i] = req
	}

	prediction := make([]float32, len(reqs))
	for i, req := range reqs {
		prediction[i] = <-req.resultCh
	}

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
}

type predictionRequest struct {
	history  [][]float32
	hand     []float32
	action   []float32
	resultCh chan float32
}

// Handles prediction requests, attempting to batch all pending requests.
func (m *TrainedLSTM) bgPredictionHandler() {
	// No buffers here because we are collecting batches and we want the batch to be as
	// large as possible until we are ready to process it.
	outputCh := make(chan *batchPredictionRequest)
	defer close(outputCh)
	go handleBatchPredictions(m.model, outputCh)

	encodeCh := make(chan []*predictionRequest)
	defer close(encodeCh)
	// Multiple encoder threads because it is NewTensor is slow
	// and we don't want to be bottlenecked on it. This will mean too-small batches
	// for the first few, but we expect that at steady-state we will be rate-limited
	// by predictions on the GPU, so the batches will still be large (just buffered).
	for i := 0; i < numEncodingWorkers; i++ {
		go handleEncoding(m.model, encodeCh, outputCh)
	}

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

func handleEncoding(model *tf.SavedModel, batchCh chan []*predictionRequest, outputCh chan *batchPredictionRequest) {
	for batch := range batchCh {
		histories := make([][][]float32, len(batch))
		hands := make([][]float32, len(batch))
		actions := make([][]float32, len(batch))
		for i, req := range batch {
			histories[i] = req.history
			hands[i] = req.hand
			actions[i] = req.action
		}

		historyTensor, err := tf.NewTensor(histories)
		if err != nil {
			glog.Fatal(err)
		}

		handTensor, err := tf.NewTensor(hands)
		if err != nil {
			glog.Fatal(err)
		}

		actionTensor, err := tf.NewTensor(actions)
		if err != nil {
			glog.Fatal(err)
		}

		outputCh <- &batchPredictionRequest{
			history: historyTensor,
			hand:    handTensor,
			action:  actionTensor,
			batch:   batch,
		}
	}
}

type batchPredictionRequest struct {
	history *tf.Tensor
	hand    *tf.Tensor
	action  *tf.Tensor
	batch   []*predictionRequest
}

func handleBatchPredictions(model *tf.SavedModel, reqCh chan *batchPredictionRequest) {
	defer model.Session.Close()
	for req := range reqCh {
		result := predictBatch(model, req.history, req.hand, req.action)
		for i, req := range req.batch {
			req.resultCh <- result[i][0]
		}

		samplesPredicted.Add(int64(len(req.batch)))
		batchesPredicted.Add(1)
	}
}

func predictBatch(model *tf.SavedModel, history, hand, action *tf.Tensor) [][]float32 {
	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("history").Output(0): history,
			model.Graph.Operation("hand").Output(0):    hand,
			model.Graph.Operation("action").Output(0):  action,
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
