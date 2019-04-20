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

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

const (
	graphTag    = "lstm"
	outputLayer = "output/BiasAdd"
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
	BatchSize              int
	OutputDir              string
	NumEncodingWorkers     int
	MaxTrainingDataWorkers int
	MaxInferenceBatchSize  int
}

// LSTM is a model for AlphaCats to be used with DeepCFR
// and implements deepcfr.Model.
type LSTM struct {
	params      Params
	iter        int
	lastWeights []string
}

func NewLSTM(p Params) *LSTM {
	return &LSTM{
		params:      p,
		iter:        1,
		lastWeights: make([]string, 2),
	}
}

// Train implements deepcfr.Model.
func (m *LSTM) Train(samples deepcfr.Buffer) deepcfr.TrainedModel {
	trainingData := samples.GetSamples()
	glog.Infof("Training network with %d samples", len(trainingData))
	// Save training data to disk in a tempdir.
	tmpDir, err := ioutil.TempDir(m.params.OutputDir, "training-data-")
	if err != nil {
		glog.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	glog.Infof("Saving training data to: %v", tmpDir)
	if err := saveTrainingData(trainingData, tmpDir, m.params.BatchSize, m.params.MaxTrainingDataWorkers); err != nil {
		glog.Fatal(err)
	}

	// Shell out to Python to train the network.
	outputDir := filepath.Join(m.params.OutputDir, fmt.Sprintf("model_%08d", m.iter))
	args := []string{"model/train.py", tmpDir, outputDir}
	initialWeights := m.lastWeights[(m.iter+1)%2]
	if initialWeights != "" {
		args = append(args, "--initial_weights", initialWeights)
	}
	cmd := exec.Command("python", args...)
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
	m.lastWeights[m.iter%2] = filepath.Join(outputDir, "weights.h5")

	// Load trained model.
	trained, err := LoadTrainedLSTM(outputDir, m.params)
	if err != nil {
		glog.Fatal(err)
	}

	return trained
}

func (m *LSTM) UnmarshalBinary(buf []byte) error {
	r := bytes.NewReader(buf)
	dec := gob.NewDecoder(r)

	if err := dec.Decode(&m.params); err != nil {
		return err
	}

	if err := dec.Decode(&m.iter); err != nil {
		return err
	}

	if err := dec.Decode(&m.lastWeights); err != nil {
		return err
	}

	return nil
}

func (m *LSTM) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(m.params); err != nil {
		return nil, err
	}

	if err := enc.Encode(m.iter); err != nil {
		return nil, err
	}

	if err := enc.Encode(m.lastWeights); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

type TrainedLSTM struct {
	dir    string
	params Params
	model  *tf.SavedModel
	reqsCh chan []*predictionRequest
}

func LoadTrainedLSTM(dir string, params Params) (*TrainedLSTM, error) {
	opts := &tf.SessionOptions{Config: tfConfig}
	model, err := tf.LoadSavedModel(dir, []string{graphTag}, opts)
	if err != nil {
		return nil, err
	}

	m := &TrainedLSTM{
		dir:    dir,
		params: params,
		model:  model,
		reqsCh: make(chan []*predictionRequest, 1),
	}
	go m.bgPredictionHandler()
	return m, nil
}

// When serializing, we just serialize the path to the model already on disk.
func (m *TrainedLSTM) GobDecode(buf []byte) error {
	r := bytes.NewReader(buf)
	dec := gob.NewDecoder(r)

	var dir string
	if err := dec.Decode(&dir); err != nil {
		return err
	}

	var params Params
	if err := dec.Decode(&params); err != nil {
		return err
	}

	model, err := LoadTrainedLSTM(dir, params)
	if err != nil {
		return err
	}

	*m = *model
	return nil
}

func (m *TrainedLSTM) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(m.dir); err != nil {
		return nil, err
	}

	if err := enc.Encode(m.params); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (m *TrainedLSTM) Close() {
	close(m.reqsCh)
}

const (
	tfHistorySize = 4 * gamestate.MaxNumActions * numActionFeatures
	tfHandSize    = 4 * cards.NumTypes
)

var predictionRequestPool = sync.Pool{
	New: func() interface{} {
		return &predictionRequest{
			action:   make([]byte, 4*numActionFeatures),
			resultCh: make(chan float32, 1),
		}
	},
}

// Predict implements deepcfr.TrainedModel.
func (m *TrainedLSTM) Predict(infoSet cfr.InfoSet, nActions int) []float32 {
	is := infoSet.(*alphacats.InfoSetWithAvailableActions)
	if len(is.AvailableActions) != nActions {
		panic(fmt.Errorf("InfoSet has %d actions but expected %d: %v",
			len(is.AvailableActions), nActions, is.AvailableActions))
	}

	tfHistory := make([]byte, tfHistorySize)
	encodeHistoryTF(is.History, tfHistory)
	tfHand := make([]byte, tfHandSize)
	encodeHandTF(is.Hand, tfHand)
	reqs := make([]*predictionRequest, nActions)
	for i, action := range is.AvailableActions {
		req := predictionRequestPool.Get().(*predictionRequest)
		req.history = tfHistory
		req.hand = tfHand
		encodeActionTF(action, req.action)
		reqs[i] = req
	}

	m.reqsCh <- reqs[:nActions]
	advantages := make([]float32, nActions)
	for i := range advantages {
		req := reqs[i]
		advantages[i] = <-req.resultCh
		req.history = nil
		req.hand = nil
		predictionRequestPool.Put(req)
	}

	return advantages
}

type predictionRequest struct {
	history  []byte
	hand     []byte
	action   []byte
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
	// Multiple encoder threads because NewTensor is slow
	// and we don't want to be bottlenecked on it. This will mean too-small batches
	// for the first few, but we expect that at steady-state we will be rate-limited
	// by predictions on the GPU, so the batches will still be large (just buffered).
	for i := 0; i < m.params.NumEncodingWorkers; i++ {
		go handleEncoding(m.model, encodeCh, outputCh)
	}

	for {
		// Wait for first request so we know we have at least one.
		batch, ok := <-m.reqsCh
		if !ok {
			return
		}

		// Drain any additional requests as long as we're waiting for an encoding spot.
	Loop:
		for {
			select {
			case reqs, ok := <-m.reqsCh:
				if !ok {
					encodeCh <- batch
					return
				}

				batch = append(batch, reqs...)
				if len(batch) >= m.params.MaxInferenceBatchSize {
					encodeCh <- batch
					break Loop
				}
			case encodeCh <- batch:
				break Loop
			}
		}
	}
}

func concat(batch []*predictionRequest) (histories, hands, actions []byte) {
	historyLen := 0
	handsLen := 0
	actionsLen := 0
	for _, req := range batch {
		historyLen += len(req.history)
		handsLen += len(req.hand)
		actionsLen += len(req.action)
	}

	histories = make([]byte, 0, historyLen)
	hands = make([]byte, 0, handsLen)
	actions = make([]byte, 0, actionsLen)
	for _, req := range batch {
		histories = append(histories, req.history...)
		hands = append(hands, req.hand...)
		actions = append(actions, req.action...)
	}

	return histories, hands, actions
}

func handleEncoding(model *tf.SavedModel, batchCh chan []*predictionRequest, outputCh chan *batchPredictionRequest) {
	for batch := range batchCh {
		// TODO: Shapes should be passed in to avoid coupling here.
		historiesBuf, handsBuf, actionsBuf := concat(batch)
		historiesReader := bytes.NewReader(historiesBuf)
		historiesShape := []int64{int64(len(batch)), gamestate.MaxNumActions, numActionFeatures}
		historyTensor, err := tf.ReadTensor(tf.Float, historiesShape, historiesReader)
		if err != nil {
			glog.Fatal(err)
		}

		handsReader := bytes.NewReader(handsBuf)
		handsShape := []int64{int64(len(batch)), int64(cards.NumTypes)}
		handTensor, err := tf.ReadTensor(tf.Float, handsShape, handsReader)
		if err != nil {
			glog.Fatal(err)
		}

		actionsReader := bytes.NewReader(actionsBuf)
		actionsShape := []int64{int64(len(batch)), numActionFeatures}
		actionTensor, err := tf.ReadTensor(tf.Float, actionsShape, actionsReader)
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

type batchResult struct {
	batch        []*predictionRequest
	resultTensor *tf.Tensor
}

func handleBatchPredictions(model *tf.SavedModel, reqCh chan *batchPredictionRequest) {
	resultsCh := make(chan batchResult, 1)
	defer close(resultsCh)
	go handleResults(resultsCh)

	defer model.Session.Close()
	for req := range reqCh {
		resultTensor := predictBatch(model, req.history, req.hand, req.action)
		resultsCh <- batchResult{req.batch, resultTensor}
		samplesPredicted.Add(int64(len(req.batch)))
		batchesPredicted.Add(1)
	}
}

func predictBatch(model *tf.SavedModel, history, hand, action *tf.Tensor) *tf.Tensor {
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

	return result[0]
}

func handleResults(resultsCh chan batchResult) {
	for batchResult := range resultsCh {
		result := batchResult.resultTensor.Value().([][]float32)
		for i, req := range batchResult.batch {
			req.resultCh <- result[i][0]
		}
	}
}

func init() {
	gob.Register(&LSTM{})
	gob.Register(&TrainedLSTM{})
}
