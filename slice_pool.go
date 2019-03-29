package alphacats

import (
	"github.com/timpalpant/alphacats/gamestate"
)

type gameNodeSlicePool struct {
	pool [][]GameNode
}

func (p *gameNodeSlicePool) alloc(n int) []GameNode {
	if p == nil {
		return make([]GameNode, 0, n)
	}

	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		return next
	}

	return make([]GameNode, 0, n)
}

func (p *gameNodeSlicePool) free(s []GameNode) {
	if p != nil && cap(s) > 0 {
		p.pool = append(p.pool, s[:0])
	}
}

type actionSlicePool struct {
	pool [][]gamestate.Action
}

func (p *actionSlicePool) alloc(n int) []gamestate.Action {
	if p == nil {
		return make([]gamestate.Action, 0, n)
	}

	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		return next
	}

	return make([]gamestate.Action, 0, n)
}

func (p *actionSlicePool) free(s []gamestate.Action) {
	if p != nil && cap(s) > 0 {
		p.pool = append(p.pool, s[:0])
	}
}
