package matrixgame

import (
	"math"
	"math/rand"

	"github.com/golang/glog"
)

func FictitiousPlay(winRateMatrix [][]float64, nIter int, mixingLambda float64) ([]float32, []float32) {
	p0PlayCounts := make([]int, len(winRateMatrix))
	p1PlayCounts := make([]int, len(winRateMatrix[0]))
	for i := 1; i <= nIter; i++ {
		var p0Selected int
		if rand.Float64() < mixingLambda {
			p0Selected = rand.Intn(len(p0PlayCounts))
		} else {
			p0Selected = getP0BestResponse(winRateMatrix, p1PlayCounts)
		}

		var p1Selected int
		if rand.Float64() < mixingLambda {
			p1Selected = rand.Intn(len(p1PlayCounts))
		} else {
			p1Selected = getP1BestResponse(winRateMatrix, p0PlayCounts)
		}
		p0PlayCounts[p0Selected] += 1
		p1PlayCounts[p1Selected] += 1

		if i%(nIter/10) == 0 {
			glog.Infof("After %d iterations, player 0 weights: %v", i, normalize(p0PlayCounts))
			glog.Infof("After %d iterations, player 1 weights: %v", i, normalize(p1PlayCounts))
		}
	}

	return normalize(p0PlayCounts), normalize(p1PlayCounts)
}

func getP0BestResponse(winRateMatrix [][]float64, p1PlayCounts []int) int {
	utilities := make([]float64, len(winRateMatrix))
	for j, c := range p1PlayCounts {
		for i := range utilities {
			utilities[i] += float64(c) * winRateMatrix[i][j]
		}
	}

	_, br := argMax(utilities)
	return br
}

func getP1BestResponse(winRateMatrix [][]float64, p0PlayCounts []int) int {
	utilities := make([]float64, len(winRateMatrix[0]))
	for i, c := range p0PlayCounts {
		for j := range utilities {
			utilities[j] -= float64(c) * winRateMatrix[i][j]
		}
	}

	_, br := argMax(utilities)
	return br
}

func normalize(counts []int) []float32 {
	total := 0
	for _, v := range counts {
		total += v
	}

	result := make([]float32, len(counts))
	for i, v := range counts {
		result[i] = float32(v) / float32(total)
	}
	return result
}

func argMax(vs []float64) (float64, int) {
	best := -math.MaxFloat64
	bestIdx := 0
	for i, v := range vs {
		if v > best {
			best = v
			bestIdx = i
		} else if v == best && rand.Intn(2) == 1 {
			bestIdx = i
		}
	}

	return best, bestIdx
}
