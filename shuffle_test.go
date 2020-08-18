package alphacats

import (
	"testing"

	"github.com/timpalpant/alphacats/cards"
)

func TestEnumerateShuffles(t *testing.T) {
	stack := cards.NewStackFromCards([]cards.Card{
		cards.Shuffle,
		cards.Skip,
		cards.Cat,
		cards.Cat,
	})

	n := CountDistinctShuffles(stack.ToSet())
	allShuffles := make([]cards.Stack, 0, n)
	EnumerateShuffles(stack.ToSet(), func(shuffle cards.Stack) {
		allShuffles = append(allShuffles, shuffle)
	})

	if len(allShuffles) != n {
		t.Errorf("expected %d shuffles, got %v", n, len(allShuffles))
	}
}

func TestNthShuffle(t *testing.T) {
	stack := cards.NewStackFromCards([]cards.Card{
		cards.Shuffle,
		cards.Skip,
		cards.Cat,
		cards.Cat,
	})

	n := factorial[stack.Len()]
	allShuffles := make([]cards.Stack, n)
	for i := 0; i < n; i++ {
		shuffle := nthShuffle(stack, i)
		t.Logf("i = %d, shuffle = %v", i, shuffle)
		allShuffles[i] = shuffle
	}
}
