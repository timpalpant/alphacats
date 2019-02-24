package model

import (
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
}

// Predict implements deepcfr.Model.
func (m *LSTM) Predict(infoSet cfr.InfoSet, nActions int) []float32 {
	p := make([]float32, nActions)
	for i := range p {
		p[i] = 1.0 / float32(nActions)
	}
	return p
}
