package alphacats

type gameNodeSlicePool struct {
	pool [][]GameNode
}

func (p *gameNodeSlicePool) alloc(n int) []GameNode {
	if p == nil {
		return make([]GameNode, n)
	}

	if len(p.pool) > 0 {
		m := len(p.pool)
		next := p.pool[m-1]
		p.pool = p.pool[:m-1]
		return append(next, make([]GameNode, n)...)
	}

	return make([]GameNode, n)
}

func (p *gameNodeSlicePool) free(s []GameNode) {
	if p != nil && cap(s) > 0 {
		p.pool = append(p.pool, s[:0])
	}
}
