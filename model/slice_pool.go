package model

import (
	"sync"

	"github.com/timpalpant/alphacats/gamestate"
)

type floatSlicePool struct {
	mx   sync.Mutex
	pool [][]float32
}

func (p *floatSlicePool) alloc(n int) []float32 {
	p.mx.Lock()

	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		p.mx.Unlock()
		return append(next, make([]float32, n)...)
	}

	p.mx.Unlock()
	return make([]float32, n)
}

func (p *floatSlicePool) free(s []float32) {
	p.mx.Lock()
	if cap(s) > 0 {
		p.pool = append(p.pool, s[:0])
	}
	p.mx.Unlock()
}

type floatHistoryPool struct {
	mx   sync.Mutex
	pool [][][]float32
}

func (p *floatHistoryPool) alloc() [][]float32 {
	p.mx.Lock()

	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		p.mx.Unlock()
		return next
	}

	p.mx.Unlock()
	return newHistorySlice()
}

func newHistorySlice() [][]float32 {
	result := make([][]float32, gamestate.MaxNumActions)
	for i := range result {
		result[i] = make([]float32, numActionFeatures)
	}
	return result
}

func (p *floatHistoryPool) free(s [][]float32) {
	p.mx.Lock()
	if cap(s) > 0 {
		p.pool = append(p.pool, s)
	}
	p.mx.Unlock()
}

type byteSlicePool struct {
	mx   sync.Mutex
	pool [][]byte
}

func (p *byteSlicePool) alloc(n int) []byte {
	p.mx.Lock()

	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		p.mx.Unlock()
		return append(next, make([]byte, n)...)
	}

	p.mx.Unlock()
	return make([]byte, n)
}

func (p *byteSlicePool) free(s []byte) {
	p.mx.Lock()
	if cap(s) > 0 {
		p.pool = append(p.pool, s[:0])
	}
	p.mx.Unlock()
}
