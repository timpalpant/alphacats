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

// Return the nth permutation of deck using factorial number system.
// See: https://en.wikipedia.org/wiki/Lehmer_code
func nthShuffle(deck cards.Stack, n int) cards.Stack {
	result := cards.NewStack()
	l := deck.Len()
	for i := 0; i < l-1; i++ {
		radix := factorial(l - i - 1)
		k := int(n / radix)
		n %= radix

		result.SetNthCard(i, deck.NthCard(k))
		deck.RemoveCard(k)
	}

	result.SetNthCard(l-1, deck.NthCard(0))
	return result
}

func factorial(k int) int {
	result := 1
	for i := 2; i <= k; i++ {
		result *= i
	}
	return result
}
