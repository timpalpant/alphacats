package alphacats

import (
	"sync"

	"github.com/timpalpant/alphacats/gamestate"
)

const maxPoolSize = 1024

type gameNodeSlicePool struct {
	mx   sync.Mutex
	pool [][]GameNode
}

func (p *gameNodeSlicePool) alloc(n int) []GameNode {
	p.mx.Lock()
	m := len(p.pool)
	if m > 0 {
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		p.mx.Unlock()
		return next[:0]
	}
	p.mx.Unlock()

	return make([]GameNode, 0, n)
}

func (p *gameNodeSlicePool) free(s []GameNode) {
	p.mx.Lock()
	if len(p.pool) < maxPoolSize {
		p.pool = append(p.pool, s)
	}
	p.mx.Unlock()
}

type actionSlicePool struct {
	mx   sync.Mutex
	pool [][]gamestate.Action
}

func (p *actionSlicePool) alloc(n int) []gamestate.Action {
	p.mx.Lock()
	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		p.mx.Unlock()
		return next[:0]
	}
	p.mx.Unlock()

	return make([]gamestate.Action, 0, n)
}

func (p *actionSlicePool) free(s []gamestate.Action) {
	p.mx.Lock()
	if len(p.pool) < maxPoolSize {
		p.pool = append(p.pool, s)
	}
	p.mx.Unlock()
}
