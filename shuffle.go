package alphacats

import (
	"github.com/timpalpant/alphacats/cards"
)

func countDistinctShuffles(deck cards.Set) int {
	result := factorial(deck.Len())
	deck.Iter(func(card cards.Card, count uint8) {
		result /= factorial(int(count))
	})
	return result
}

func enumerateShuffles(deck cards.Set, cb func(shuffle cards.Stack)) {
	enumerateShufflesHelper(deck, cards.NewStack(), 0, cb)
}

func enumerateShufflesHelper(deck cards.Set, result cards.Stack, n int, cb func(shuffle cards.Stack)) {
	if deck.IsEmpty() { // All cards have been used, complete shuffle.
		cb(result)
		return
	}

	deck.Iter(func(card cards.Card, count uint8) {
		// Take one of card from deck and append to result.
		remaining := deck
		remaining.Remove(card)
		newResult := result
		newResult.InsertCard(card, n)
		// Recurse with remaining deck and new result.
		enumerateShufflesHelper(remaining, newResult, n+1, cb)
	})
}

func nthShuffle(drawPile cards.Stack, i int) cards.Stack {
	result := cards.NewStack()
	code := lehmerCode(drawPile.Len(), i)
	for i, k := range code {
		result.SetNthCard(i, drawPile.NthCard(k))
		drawPile.RemoveCard(k)
	}
	return result
}

// Return the kth permutation of n items using factorial number system.
// See: https://en.wikipedia.org/wiki/Lehmer_code
func lehmerCode(n, k int) []int {
	if n <= 1 {
		return []int{0}
	}

	radix := factorial(n - 1)
	digit := int(k / radix)
	remainder := k % radix
	return append([]int{digit}, lehmerCode(n-1, remainder)...)
}

func factorial(k int) int {
	result := 1
	for i := 2; i <= k; i++ {
		result *= i
	}
	return result
}
