package model

import (
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

const graphTag = "lstm"

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

type TrainedLSTM struct {
	dir   string
	model *tf.SavedModel
}

func LoadTrainedLSTM(dir string) (*TrainedLSTM, error) {
	opts := &tf.SessionOptions{Config: tfConfig}
	model, err := tf.LoadSavedModel(dir, []string{graphTag}, opts)
	if err != nil {
		return nil, err
	}

	return &TrainedLSTM{dir, model}, nil
}

// Predict implements deepcfr.TrainedModel.
func (m *TrainedLSTM) Predict(infoSet cfr.InfoSet, nActions int) []float32 {
	is := infoSet.(*gamestate.InfoSet)
	history := EncodeHistory(is.History)
	historyTensor, err := tf.NewTensor([][][]float32{history})
	if err != nil {
		glog.Fatal(err)
	}

	hand := encodeHand(is.Hand)
	handTensor, err := tf.NewTensor([][]float32{hand})
	if err != nil {
		glog.Fatal(err)
	}

	result, err := m.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.model.Graph.Operation("history").Output(0): historyTensor,
			m.model.Graph.Operation("hand").Output(0):    handTensor,
		},
		[]tf.Output{
			m.model.Graph.Operation("output/BiasAdd").Output(0),
		},
		nil,
	)

	if err != nil {
		glog.Fatal(err)
	}

	prediction := result[0].Value().([][]float32)
	glog.V(1).Infof("Predicted advantages: %v", prediction)

	advantages := prediction[0][:nActions]
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
