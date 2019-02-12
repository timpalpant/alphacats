package alphacats

import (
	"sync"
)

var (
	gameNodeSlicePool = sync.Pool{
		New: func() interface{} {
			return make([]GameNode, 0)
		},
	}

	floatSlicePool = sync.Pool{
		New: func() interface{} {
			return make([]float64, 0)
		},
	}
)

func allocGameNodeSlice() []GameNode {
	return gameNodeSlicePool.Get().([]GameNode)
}

func freeGameNodeSlice(s []GameNode) {
	if cap(s) > 0 {
		gameNodeSlicePool.Put(s[:0])
	}
}

func allocFloatSlice() []float64 {
	return floatSlicePool.Get().([]float64)
}

func freeFloatSlice(s []float64) {
	if cap(s) > 0 {
		floatSlicePool.Put(s[:0])
	}
}
