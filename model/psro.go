package model

import (
	"bytes"
	"encoding/gob"
	"expvar"
	"io"
	"math/rand"
	"sync"
	"sync/atomic"

	"github.com/golang/glog"
	"github.com/hashicorp/golang-lru"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/mcts"
	"github.com/timpalpant/go-cfr/sampling"

	"github.com/timpalpant/alphacats"
)

var (
	cacheHits    = expvar.NewInt("predictions/cache_hits")
	cacheMisses  = expvar.NewInt("predictions/cache_misses")
	cacheHitRate = expvar.NewFloat("predictions/cache_hit_rate")
	cacheSize    = expvar.NewInt("predictions/cache_size")
)

type RefCountingTrainedLSTM struct {
	*TrainedLSTM
	RefCount int64
}

type MCTSPSRO struct {
	model               *LSTM
	predictionCacheSize int

	policies []mcts.Policy
	weights  []float32

	mx         sync.Mutex
	samples    []Sample
	maxSamples int
	sampleIdx  int

	currentNetwork *RefCountingTrainedLSTM
	rollout        mcts.Evaluator
}

func NewMCTSPSRO(model *LSTM, maxSamples, predictionCacheSize int) *MCTSPSRO {
	return &MCTSPSRO{
		model:               model,
		predictionCacheSize: predictionCacheSize,
		policies:            []mcts.Policy{},
		weights:             []float32{},
		samples:             make([]Sample, 0, maxSamples),
		maxSamples:          maxSamples,
		rollout:             mcts.NewRandomRollout(1),
	}
}

func LoadMCTSPSRO(r io.Reader) (*MCTSPSRO, error) {
	m := &MCTSPSRO{
		rollout: mcts.NewRandomRollout(1),
	}
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&m.model); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.predictionCacheSize); err != nil {
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
	if err := dec.Decode(&m.maxSamples); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.sampleIdx); err != nil {
		return nil, err
	}
	if err := dec.Decode(&m.currentNetwork); err != nil {
		glog.Warningf("Did not load in-progress network: %v", err)
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
	if err := enc.Encode(m.predictionCacheSize); err != nil {
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
	if err := enc.Encode(m.maxSamples); err != nil {
		return err
	}
	if err := enc.Encode(m.sampleIdx); err != nil {
		return err
	}
	if m.currentNetwork != nil {
		if err := enc.Encode(m.currentNetwork); err != nil {
			return err
		}
	}
	return nil
}

func (m *MCTSPSRO) AddSample(s Sample) {
	m.mx.Lock()
	defer m.mx.Unlock()
	if len(m.samples) < m.maxSamples {
		m.samples = append(m.samples, s)
	} else {
		m.samples[m.sampleIdx%m.maxSamples] = s
	}

	m.sampleIdx++
}

func (m *MCTSPSRO) TrainNetwork() {
	m.mx.Lock()
	defer m.mx.Unlock()

	var initialWeightsFile string
	if m.currentNetwork != nil {
		initialWeightsFile = m.currentNetwork.KerasWeightsFile()
	}

	// TODO(palpant): Unlock to allow other games to continue playing while retraining.
	nn := m.model.Train(initialWeightsFile, m.samples)
	// TODO(palpant): Implement evaluation/selection by pitting this network
	// against the previous best response network and only keeping it if it wins
	// at least 55% of the time.
	if m.currentNetwork != nil {
		if m.currentNetwork.RefCount == 1 {
			// There are no other concurrent uses of this network making predictions, close it.
			m.currentNetwork.Close()
		} else {
			// There are other concurrent uses of this network.
			// Decrement the reference count so that when the last one finishes, it is closed.
			atomic.AddInt64(&m.currentNetwork.RefCount, -1)
		}
	}
	m.currentNetwork = &RefCountingTrainedLSTM{nn, 1}
}

func (m *MCTSPSRO) AddCurrentExploiterToModel() {
	m.mx.Lock()
	defer m.mx.Unlock()
	if m.currentNetwork == nil {
		return
	}

	// TODO(palpant): Implement real PSRO. For now we just average all networks
	// with equal weight (aka Fictitious Play).
	m.policies = append(m.policies, NewPredictorPolicy(m.currentNetwork.TrainedLSTM, m.predictionCacheSize))
	m.weights = uniformDistribution(len(m.policies))
	m.currentNetwork = nil
	m.samples = m.samples[:0]
	m.sampleIdx = 0
	glog.Infof("Added network. PSRO now has %d oracles", len(m.policies))
}

func (m *MCTSPSRO) AddModel(policy mcts.Policy) {
	m.mx.Lock()
	defer m.mx.Unlock()
	m.policies = append(m.policies, policy)
	m.weights = uniformDistribution(len(m.policies))
}

func (m *MCTSPSRO) Len() int {
	return len(m.policies)
}

func (m *MCTSPSRO) SamplePolicy() mcts.Policy {
	selected := sampling.SampleOne(m.weights, rand.Float32())
	return m.policies[selected]
}

// Evaluate implements mcts.Evaluator for one-sided IS-MCTS search rollouts
// when this policy is being trained as the exploiter.
func (m *MCTSPSRO) Evaluate(rng *rand.Rand, node cfr.GameTreeNode, opponent mcts.Policy) ([]float32, float32) {
	nn := m.getCurrentNetwork()
	if nn == nil {
		return m.rollout.Evaluate(rng, node, opponent)
	}
	defer m.releaseNetwork(nn)

	is := node.InfoSet(node.Player()).(*alphacats.AbstractedInfoSet)
	return nn.Predict(is)
}

func (m *MCTSPSRO) getCurrentNetwork() *RefCountingTrainedLSTM {
	m.mx.Lock()
	defer m.mx.Unlock()
	if m.currentNetwork != nil {
		atomic.AddInt64(&m.currentNetwork.RefCount, 1)
	}
	return m.currentNetwork
}

func (m *MCTSPSRO) releaseNetwork(nn *RefCountingTrainedLSTM) {
	numRefs := atomic.AddInt64(&nn.RefCount, -1)
	if numRefs == 0 {
		// This is the last reference to this network, close it.
		nn.Close()
	}
}

// Policy that always plays randomly. Used to bootstrap fictitious play.
// Alternative to SmoothUCT for bootstrapping fictitious play.
type UniformRandomPolicy struct{}

func (u *UniformRandomPolicy) GetPolicy(node cfr.GameTreeNode) []float32 {
	return uniformDistribution(node.NumChildren())
}

// Adaptor to use TrainedLSTM as a mcts.Policy.
type PredictorPolicy struct {
	model     *TrainedLSTM
	cache     *lru.Cache
	cacheSize int
}

func NewPredictorPolicy(model *TrainedLSTM, cacheSize int) *PredictorPolicy {
	cache, err := lru.New(cacheSize)
	if err != nil {
		panic(err)
	}

	return &PredictorPolicy{
		model:     model,
		cache:     cache,
		cacheSize: cacheSize,
	}
}

func (pp *PredictorPolicy) GetPolicy(node cfr.GameTreeNode) []float32 {
	is := node.InfoSet(node.Player()).(*alphacats.AbstractedInfoSet)
	key := string(is.Key())
	cached, ok := pp.cache.Get(key)
	if ok {
		cacheHits.Add(1)
		cacheHitRate.Set(float64(cacheHits.Value()) / float64(cacheHits.Value()+cacheMisses.Value()))
		return cached.([]float32)
	}

	cacheMisses.Add(1)
	cacheHitRate.Set(float64(cacheHits.Value()) / float64(cacheHits.Value()+cacheMisses.Value()))
	p, _ := pp.model.Predict(is)
	pp.cache.Add(key, p)
	cacheSize.Set(int64(pp.cache.Len()))
	return p
}

func (pp *PredictorPolicy) GobDecode(buf []byte) error {
	r := bytes.NewReader(buf)
	dec := gob.NewDecoder(r)

	if err := dec.Decode(&pp.model); err != nil {
		return err
	}

	if err := dec.Decode(&pp.cacheSize); err != nil {
		return err
	}

	cache, err := lru.New(pp.cacheSize)
	if err != nil {
		return err
	}
	pp.cache = cache

	return nil
}

func (pp *PredictorPolicy) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(pp.model); err != nil {
		return nil, err
	}
	if err := enc.Encode(pp.cacheSize); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func uniformDistribution(n int) []float32 {
	result := make([]float32, n)
	for i := range result {
		result[i] = 1.0 / float32(n)
	}
	return result
}

func init() {
	gob.Register(&UniformRandomPolicy{})
	gob.Register(&PredictorPolicy{})
	gob.Register(&RefCountingTrainedLSTM{})
}
