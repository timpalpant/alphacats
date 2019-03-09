package model

import (
	"io/ioutil"
	"os"
	"os/exec"

	"github.com/golang/glog"

	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"
)

const trainingScript = "../../model/run_training.sh"

type Params struct {
	BatchSize            int
	Optimizer            string
	LearningRate         float64
	GradientNormClipping float64
}

func DefaultParams() Params {
	return Params{
		BatchSize:            1000000,
		Optimizer:            "adam",
		LearningRate:         0.001,
		GradientNormClipping: 10.0,
	}
}

// LSTM is a model for AlphaCats to be used with DeepCFR.
type LSTM struct {
	params Params
}

func NewLSTM(p Params) *LSTM {
	return &LSTM{p}
}

// Train implements deepcfr.Model.
func (m *LSTM) Train(samples deepcfr.Buffer) {
	glog.Infof("Training network with %d samples", len(samples.GetSamples()))
	// Save training data to disk.
	tmpDir, err := ioutil.TempDir("", "alphacats-training-data-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(tmpDir)

	glog.Infof("Saving training data to: %v", tmpDir)
	if err := saveTrainingData(samples.GetSamples(), tmpDir, m.params.BatchSize); err != nil {
		panic(err)
	}

	// Shell out to Python for training.
	cmd := exec.Command(trainingScript, tmpDir, "model.hd5")
	if err := cmd.Run(); err != nil {
		panic(err)
	}

	// Load trained model.
	// https://github.com/galeone/tfgo#tfgo-tensorflow-in-go
}

// Predict implements deepcfr.Model.
func (m *LSTM) Predict(infoSet cfr.InfoSet, nActions int) []float32 {
	p := make([]float32, nActions)
	for i := range p {
		p[i] = 1.0 / float32(nActions)
	}
	return p
}
