package model

import (
	"sync"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
)

type MCTSPSRO struct {
	model           *LSTM
	retrainInterval int

	bestResponseNetworks []*TrainedLSTM
	weights              []float32

	mx        sync.Mutex
	samples   []Sample
	sampleIdx int

	currentNetwork *TrainedLSTM
	needsRetrain   bool
}

func NewMCTSPSRO(model *LSTM, maxSamples, maxSampleReuse int) *MCTSPSRO {
	return &MCTSPSRO{
		model:           model,
		retrainInterval: maxSamples / maxSampleReuse,
		samples:         make([]Sample, 0, maxSamples),
	}
}

func (m *MCTSPSRO) AddSample(s Sample) {
	m.mx.Lock()
	defer m.mx.Unlock()
	if len(m.samples) < cap(m.samples) {
		m.samples = append(m.samples, s)
	} else {
		m.samples[m.sampleIdx%len(m.samples)] = s
	}

	m.sampleIdx++
	if m.sampleIdx%m.retrainInterval == 0 {
		m.needsRetrain = true
	}
}

func (m *MCTSPSRO) TrainNetwork() {
	m.mx.Lock()
	defer m.mx.Unlock()
	if !m.needsRetrain {
		need := m.retrainInterval - (m.sampleIdx % m.retrainInterval)
		glog.Infof("Need %d more samples before retraining", need)
		return // Not enough data to retrain yet.
	}

	var initialWeightsFile string
	if m.currentNetwork != nil {
		initialWeightsFile = m.currentNetwork.KerasWeightsFile()
	}

	// TODO(palpant): Unlock to allow other games to continue playing while retraining.
	nn := m.model.Train(initialWeightsFile, m.samples)
	// TODO(palpant): Implement evaluation/selection by pitting this network
	// against the previous best response network and only keeping it if it wins
	// at least 55% of the time.
	m.currentNetwork = nn
	m.needsRetrain = false
}

func (m *MCTSPSRO) AddCurrentExploiterToModel() {
	m.mx.Lock()
	defer m.mx.Unlock()
	// TODO(palpant): Implement real PSRO. For now we just average all networks
	// with equal weight (aka Fictitious Play).
	m.bestResponseNetworks = append(m.bestResponseNetworks, m.currentNetwork)
	m.weights = uniformDistribution(len(m.bestResponseNetworks))
	m.currentNetwork = nil
	m.samples = m.samples[:0]
	m.sampleIdx = 0
	m.needsRetrain = false
	glog.Infof("Added network. PSRO now has %d oracles", len(m.bestResponseNetworks))
}

// GetPolicy implements mcts.Policy for one-sided IS-MCTS search when this policy is
// the (fixed) opponent.
func (m *MCTSPSRO) GetPolicy(node cfr.GameTreeNode) []float32 {
	if len(m.bestResponseNetworks) == 0 {
		return uniformDistribution(node.NumChildren())
	}

	is := node.InfoSet(node.Player()).(*alphacats.AbstractedInfoSet)
	result := make([]float32, node.NumChildren())
	var wg sync.WaitGroup
	var mx sync.Mutex
	for i, nn := range m.bestResponseNetworks {
		wg.Add(1)
		go func(i int, nn *TrainedLSTM) {
			defer wg.Done()
			p, _ := nn.Predict(is)
			mx.Lock()
			defer mx.Unlock()
			for i, pi := range p {
				result[i] += m.weights[i] * pi
			}
		}(i, nn)
	}

	wg.Wait()
	normalize(result)
	return result
}

// Evaluate implements mcts.Evaluator for one-sided IS-MCTS search rollouts
// when this policy is being trained as the exploiter.
func (m *MCTSPSRO) Evaluate(node cfr.GameTreeNode) ([]float32, float32) {
	nn := m.getCurrentNetwork()
	if nn == nil {
		p := uniformDistribution(node.NumChildren())
		v := float32(0.0)
		return p, v
	}

	is := node.InfoSet(node.Player()).(*alphacats.AbstractedInfoSet)
	return nn.Predict(is)
}

func (m *MCTSPSRO) getCurrentNetwork() *TrainedLSTM {
	m.mx.Lock()
	defer m.mx.Unlock()
	return m.currentNetwork
}

func uniformDistribution(n int) []float32 {
	result := make([]float32, n)
	for i := range result {
		result[i] = 1.0 / float32(n)
	}
	return result
}
