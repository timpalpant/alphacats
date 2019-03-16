// Package model implements an LSTM-based network model for use in DeepCFR.
// As input, the model takes the public game history (one-hot encoded),
// as well as the player's current hand (one-hot encoded) and predicts the
// advantages for each possible action in this infoset.
package model

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"

	"github.com/golang/glog"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats/gamestate"
)

const (
	graphTag               = "lstm"
	maxPredictionBatchSize = 256
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
	if err := saveTrainingData(samples.GetSamples(), tmpDir, m.params.BatchSize); err != nil {
		glog.Fatal(err)
	}

	// Shell out to Python to train the network.
	outputDir := fmt.Sprintf("model_%08d", m.iter)
	cmd := exec.Command("python", "model/train.py", tmpDir, outputDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	glog.Infof("Running: %v", cmd.Args)
	if err := cmd.Run(); err != nil {
		glog.Fatal(err)
	}

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
	reqCh chan predictionRequest
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
		reqCh: make(chan predictionRequest, 1),
	}
	go m.bgPredictionHandler()
	return m, nil
}

// Predict implements deepcfr.TrainedModel.
func (m *TrainedLSTM) Predict(infoSet cfr.InfoSet, nActions int) []float32 {
	is := infoSet.(*gamestate.InfoSet)
	req := predictionRequest{
		history:  EncodeHistory(is.History),
		hand:     encodeHand(is.Hand),
		resultCh: make(chan []float32),
	}

	m.reqCh <- req
	prediction := <-req.resultCh
	glog.V(1).Infof("Predicted advantages: %v", prediction)
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
	resultCh chan []float32
}

// Handles prediction requests, attempting to batch all pending requests.
func (m *TrainedLSTM) bgPredictionHandler() {
	var batch []predictionRequest

	for {
		batch, closed := drainPendingRequests(m.reqCh, batch)
		if len(batch) > 0 {
			predictBatch(m.model, batch)
			batch = batch[:0]
		}

		if closed {
			return
		}
	}
}

// Drain channel, collecting all pending prediction requests.
func drainPendingRequests(reqCh chan predictionRequest, batch []predictionRequest) ([]predictionRequest, bool) {
	for {
		select {
		case req, ok := <-reqCh:
			if !ok {
				return batch, true
			}

			batch = append(batch, req)
			if len(batch) >= maxPredictionBatchSize {
				return batch, false
			}
		default:
			return batch, false
		}
	}
}

func predictBatch(model *tf.SavedModel, batch []predictionRequest) {
	histories := make([][][]float32, len(batch))
	hands := make([][]float32, len(batch))
	for i, req := range batch {
		histories[i] = req.history
		hands[i] = req.hand
	}

	historyTensor, err := tf.NewTensor(histories)
	if err != nil {
		glog.Fatal(err)
	}

	handTensor, err := tf.NewTensor(hands)
	if err != nil {
		glog.Fatal(err)
	}

	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("history").Output(0): historyTensor,
			model.Graph.Operation("hand").Output(0):    handTensor,
		},
		[]tf.Output{
			model.Graph.Operation("output/BiasAdd").Output(0),
		},
		nil,
	)

	if err != nil {
		glog.Fatal(err)
	}

	predictions := result[0].Value().([][]float32)
	for i, req := range batch {
		req.resultCh <- predictions[i]
	}
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
