package alphacats

type gameNodeSlicePool struct {
	pool [][]GameNode
}

func (p *gameNodeSlicePool) alloc() []GameNode {
	if p == nil {
		return make([]GameNode, 0)
	}

	if len(p.pool) > 0 {
		n := len(p.pool)
		next := p.pool[n-1]
		p.pool = p.pool[:n-1]
		return next
	}

	return make([]GameNode, 0)
}

func (p *gameNodeSlicePool) free(s []GameNode) {
	if p != nil && cap(s) > 0 {
		p.pool = append(p.pool, s[:0])
	}
}

type floatSlicePool struct {
	pool [][]float64
}

func (p *floatSlicePool) alloc() []float64 {
	if p == nil {
		return make([]float64, 0)
	}

	if len(p.pool) > 0 {
		n := len(p.pool)
		next := p.pool[n-1]
		p.pool = p.pool[:n-1]
		return next
	}

	return make([]float64, 0)
}

func (p *floatSlicePool) free(s []float64) {
	if p != nil && cap(s) > 0 {
		p.pool = append(p.pool, s[:0])
	}
}
