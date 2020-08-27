package alphacats

import (
	"sync"

	"github.com/timpalpant/alphacats/gamestate"
)

type gameNodeSlicePool struct {
	mx   sync.Mutex
	pool [][]GameNode
}

func (p *gameNodeSlicePool) alloc(n int) []GameNode {
	p.mx.Lock()
	defer p.mx.Unlock()
	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		return next
	}

	return make([]GameNode, 0, n)
}

func (p *gameNodeSlicePool) free(s []GameNode) {
	if cap(s) > 0 {
		p.mx.Lock()
		p.pool = append(p.pool, s[:0])
		p.mx.Unlock()
	}
}

type actionSlicePool struct {
	mx   sync.Mutex
	pool [][]gamestate.Action
}

func (p *actionSlicePool) alloc(n int) []gamestate.Action {
	p.mx.Lock()
	defer p.mx.Unlock()
	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		return next
	}

	return make([]gamestate.Action, 0, n)
}

func (p *actionSlicePool) free(s []gamestate.Action) {
	if cap(s) > 0 {
		p.mx.Lock()
		p.pool = append(p.pool, s[:0])
		p.mx.Unlock()
	}
}
