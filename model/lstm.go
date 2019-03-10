package model

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"

	"github.com/golang/glog"

	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"
)

const trainingScript = "../../model/run_training.sh"

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
		panic(err)
	}
	defer os.RemoveAll(tmpDir)

	glog.Infof("Saving training data to: %v", tmpDir)
	if err := saveTrainingData(samples.GetSamples(), tmpDir, m.params.BatchSize); err != nil {
		panic(err)
	}

	// Shell out to Python to train the network.
	outputFilename := fmt.Sprintf("model_%08d.hd5", m.iter)
	cmd := exec.Command(trainingScript, tmpDir, outputFilename)
	if err := cmd.Run(); err != nil {
		panic(err)
	}

	m.iter++

	// Load trained model.
	// https://github.com/galeone/tfgo#tfgo-tensorflow-in-go
	return &TrainedLSTM{}
}

type TrainedLSTM struct {
}

// Predict implements deepcfr.TrainedModel.
func (m *TrainedLSTM) Predict(infoSet cfr.InfoSet, nActions int) []float32 {
	p := make([]float32, nActions)
	for i := range p {
		p[i] = 1.0 / float32(nActions)
	}
	return p
}
