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

	n := countDistinctShuffles(stack.ToSet())
	allShuffles := make([]cards.Stack, 0, n)
	enumerateShuffles(stack.ToSet(), func(shuffle cards.Stack) {
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

	n := factorial(stack.Len())
	allShuffles := make([]cards.Stack, n)
	for i := 0; i < n; i++ {
		code := lehmerCode(stack.Len(), i)
		shuffle := nthShuffle(stack, i)
		t.Logf("i = %d, code = %v, shuffle = %v", i, code, shuffle)
		allShuffles[i] = shuffle
	}
}
