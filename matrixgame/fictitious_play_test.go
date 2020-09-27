package matrixgame

import (
	"testing"
)

func TestFictitiousPlay_RockPaperScissors(t *testing.T) {
	winRateMatrix := [][]float64{
		[]float64{0, 1, -1}, // Player 0 plays rock.
		[]float64{-1, 0, 1}, // Player 0 plays scissors.
		[]float64{1, -1, 0}, // Player 0 plays paper.
	}

	p0, p1 := FictitiousPlay(winRateMatrix, 10000)
	t.Logf("Player 0 Nash equilibrium policy: %v", p0)
	t.Logf("Player 1 Nash equilibrium policy: %v", p1)
}
