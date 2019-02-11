package alphacats

var (
	gameNodeSlicePool = [][]GameNode{}
	floatSlicePool    = [][]float64{}
)

func allocGameNodeSlice() []GameNode {
	if len(gameNodeSlicePool) > 0 {
		result := gameNodeSlicePool[len(gameNodeSlicePool)-1]
		gameNodeSlicePool = gameNodeSlicePool[:len(gameNodeSlicePool)-1]
		return result
	}

	result := make([]GameNode, 0)
	return result
}

func freeGameNodeSlice(s []GameNode) {
	if cap(s) > 0 {
		gameNodeSlicePool = append(gameNodeSlicePool, s[:0])
	}
}

func allocFloatSlice() []float64 {
	if len(floatSlicePool) > 0 {
		result := floatSlicePool[len(floatSlicePool)-1]
		floatSlicePool = floatSlicePool[:len(floatSlicePool)-1]
		return result
	}

	result := make([]float64, 0)
	return result
}

func freeFloatSlice(s []float64) {
	if cap(s) > 0 {
		floatSlicePool = append(floatSlicePool, s[:0])
	}
}
