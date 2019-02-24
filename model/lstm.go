package model

import (
	"os"

	"github.com/golang/glog"
	gzip "github.com/klauspost/pgzip"

	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"
)

type Params struct {
	BatchSize            int
	Optimizer            string
	LearningRate         float64
	GradientNormClipping float64
}

func DefaultParams() Params {
	return Params{
		BatchSize:            10240,
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
	// Save training data to file.
	tmpFile, err := ioutil.TempFile("", "alphacats-training-data-")
	if err != nil {
		panic(err)
	}
	defer tmpFile.Close()
	defer os.Remove(tmpFile.Name())

	glog.Infof("Saving training data to: %v", tmpFile.Name())
	gzw := gzip.NewWriter(tmpFile)
	defer gzw.Close()
	if err := saveTrainingData(samples.GetSamples(), gzw); err != nil {
		panic(err)
	}
	gzw.Close()
	tmpFile.Close()

	// Shell out to Python for training.
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
