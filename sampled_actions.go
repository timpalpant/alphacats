package alphacats

import (
	"crypto/md5"
	"math/rand"

	"github.com/timpalpant/go-cfr"
)

type sampledActionsPool struct {
	pool    []sampledActionsMap
	bufPool *byteSlicePool
}

func newSampledActionsPool() *sampledActionsPool {
	return &sampledActionsPool{
		pool:    make([]sampledActionsMap, 0),
		bufPool: &byteSlicePool{},
	}
}

func (p *sampledActionsPool) Alloc() cfr.SampledActions {
	if len(p.pool) > 0 {
		n := len(p.pool)
		next := p.pool[n-1]
		p.pool = p.pool[:n-1]
		return next
	}

	return sampledActionsMap{
		p: p,
		m: make(map[[md5.Size]byte]uint8),
	}
}

func (p *sampledActionsPool) Free(m sampledActionsMap) {
	p.pool = append(p.pool, m)
}

// sampledActionsMap is a customized version of the one provided in go-cfr
// that stores selected actions as uint8 and keyed by md5 hash.
// Since we know we never have more available actions than fit in a uint8, it allows
// us to reduce memory usage and GC pressure (map without pointers does not get GC scanned).
type sampledActionsMap struct {
	p *sampledActionsPool
	m map[[md5.Size]byte]uint8
}

func (m sampledActionsMap) Get(node cfr.GameTreeNode, policy cfr.NodePolicy) int {
	is := node.InfoSet(node.Player()).(*InfoSetWithAvailableActions)
	buf := m.p.bufPool.alloc(is.History.Len() + 8) // Minimum possible capacity needed.
	buf, _ = is.MarshalTo(buf)
	defer m.p.bufPool.free(buf)

	key := md5.Sum(buf)
	i, ok := m.m[key]
	if !ok {
		i = sampleOne(policy.GetStrategy())
		m.m[key] = i
	}

	return int(i)
}

func (m sampledActionsMap) Close() error {
	for k := range m.m {
		delete(m.m, k)
	}

	m.p.Free(m)
	return nil
}

const eps = 1e-3

func sampleOne(pv []float32) uint8 {
	x := rand.Float32()
	var cumProb float32
	for i, p := range pv {
		cumProb += p
		if cumProb > x {
			return uint8(i)
		}
	}

	if cumProb < 1.0-eps { // Leave room for floating point error.
		panic("probability distribution does not sum to 1!")
	}

	return uint8(len(pv) - 1)
}
