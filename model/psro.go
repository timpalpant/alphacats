package model

import (
	"encoding/gob"
	"io"
	"sync"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/mcts"

	"github.com/timpalpant/alphacats"
)

type MCTSPSRO struct {
	model           *LSTM
	retrainInterval int

	policies []mcts.Policy
	weights  []float32

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
		policies:        []mcts.Policy{&UniformRandomPolicy{}},
		weights:         []float32{1.0},
		samples:         make([]Sample, 0, maxSamples),
	}
}

func LoadMCTSPSRO(r io.Reader) (*MCTSPSRO, error) {
	m := &MCTSPSRO{}
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&m.model); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.retrainInterval); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.policies); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.weights); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.samples); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.sampleIdx); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.currentNetwork); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.needsRetrain); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *MCTSPSRO) SaveTo(w io.Writer) error {
	m.mx.Lock()
	defer m.mx.Unlock()
	enc := gob.NewEncoder(w)
	if err := enc.Encode(m.model); err != nil {
		return err
	}
	if err := enc.Encode(m.retrainInterval); err != nil {
		return err
	}
	if err := enc.Encode(m.policies); err != nil {
		return err
	}
	if err := enc.Encode(m.weights); err != nil {
		return err
	}
	if err := enc.Encode(m.samples); err != nil {
		return err
	}
	if err := enc.Encode(m.sampleIdx); err != nil {
		return err
	}
	if err := enc.Encode(m.currentNetwork); err != nil {
		return err
	}
	if err := enc.Encode(m.needsRetrain); err != nil {
		return err
	}
	return nil
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
	m.policies = append(m.policies, &PredictorPolicy{m.currentNetwork})
	m.weights = uniformDistribution(len(m.policies))
	m.currentNetwork = nil
	m.samples = m.samples[:0]
	m.sampleIdx = 0
	m.needsRetrain = false
	glog.Infof("Added network. PSRO now has %d oracles", len(m.policies))
}

// GetPolicy implements mcts.Policy for one-sided IS-MCTS search when this policy is
// the (fixed) opponent.
func (m *MCTSPSRO) GetPolicy(node cfr.GameTreeNode) []float32 {
	result := make([]float32, node.NumChildren())
	var wg sync.WaitGroup
	var mx sync.Mutex
	for i, policy := range m.policies {
		wg.Add(1)
		go func(i int, policy mcts.Policy) {
			defer wg.Done()
			p := policy.GetPolicy(node)
			mx.Lock()
			defer mx.Unlock()
			for j, pj := range p {
				result[j] += m.weights[i] * pj
			}
		}(i, policy)
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

// Policy that always plays randomly. Used to bootstrap fictitious play.
type UniformRandomPolicy struct{}

func (u *UniformRandomPolicy) GetPolicy(node cfr.GameTreeNode) []float32 {
	return uniformDistribution(node.NumChildren())
}

// Adaptor to use TrainedLSTM as a mcts.Policy.
type PredictorPolicy struct {
	model *TrainedLSTM
}

func (pp *PredictorPolicy) GetPolicy(node cfr.GameTreeNode) []float32 {
	is := node.InfoSet(node.Player()).(*alphacats.AbstractedInfoSet)
	p, _ := pp.model.Predict(is)
	return p
}

func uniformDistribution(n int) []float32 {
	result := make([]float32, n)
	for i := range result {
		result[i] = 1.0 / float32(n)
	}
	return result
}
