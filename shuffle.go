package alphacats

import (
	"math/rand"

	"github.com/timpalpant/alphacats/cards"
)

type Deal struct {
	DrawPile cards.Stack
	P0Deal   cards.Set
	P1Deal   cards.Set
}

func NewRandomDeal(deck []cards.Card, cardsPerPlayer int) Deal {
	rand.Shuffle(len(deck), func(i, j int) {
		deck[i], deck[j] = deck[j], deck[i]
	})

	p0Deal := cards.NewSetFromCards(deck[:cardsPerPlayer])
	p0Deal.Add(cards.Defuse)
	p1Deal := cards.NewSetFromCards(deck[cardsPerPlayer : 2*cardsPerPlayer])
	p1Deal.Add(cards.Defuse)
	drawPile := cards.NewStackFromCards(deck[2*cardsPerPlayer:])
	randPos := rand.Intn(drawPile.Len() + 1)
	drawPile.InsertCard(cards.ExplodingKitten, randPos)
	randPos = rand.Intn(drawPile.Len() + 1)
	drawPile.InsertCard(cards.Defuse, randPos)

	return Deal{drawPile, p0Deal, p1Deal}
}

func NewRandomDealWithConstraints(drawPile cards.Stack, p1Hand cards.Set) Deal {
	p1Hand.Remove(cards.Defuse)
	remaining := cards.CoreDeck
	remaining.RemoveAll(p1Hand)
	remaining.Add(cards.Defuse)
	remaining.Add(cards.ExplodingKitten)
	for i := 0; i < drawPile.Len(); i++ {
		nthCard := drawPile.NthCard(i)
		if remaining.Contains(nthCard) {
			remaining.Remove(nthCard)
		}
	}

	hasExplodingKitten := remaining.Contains(cards.ExplodingKitten)
	if hasExplodingKitten {
		remaining.Remove(cards.ExplodingKitten)
	}
	hasDefuse := remaining.Contains(cards.Defuse)
	if hasDefuse {
		remaining.Remove(cards.Defuse)
	}

	r := remaining.AsSlice()
	rand.Shuffle(len(r), func(i, j int) {
		r[i], r[j] = r[j], r[i]
	})
	p0Hand := cards.NewSetFromCards(r[:p1Hand.Len()])

	r = r[p1Hand.Len():]
	if hasExplodingKitten {
		r = append(r, cards.ExplodingKitten)
	}
	if hasDefuse {
		r = append(r, cards.Defuse)
	}
	if hasExplodingKitten || hasDefuse {
		rand.Shuffle(len(r), func(i, j int) {
			r[i], r[j] = r[j], r[i]
		})
	}

	finalDrawPile := drawPile
	nCardsInDrawPile := cards.CoreDeck.Len() - p0Hand.Len() - p1Hand.Len() + 2
	for i := 0; i < nCardsInDrawPile; i++ {
		nthCard := finalDrawPile.NthCard(i)
		if nthCard == cards.Unknown {
			finalDrawPile.SetNthCard(i, r[0])
			r = r[1:]
		}
	}

	p0Hand.Add(cards.Defuse)
	p1Hand.Add(cards.Defuse)
	return Deal{finalDrawPile, p0Hand, p1Hand}
}

func remove(r []cards.Card, toRemove cards.Card) []cards.Card {
	var candidates []int
	for i, c := range r {
		if c == toRemove {
			candidates = append(candidates, i)
		}
	}

	selected := rand.Intn(len(candidates))
	return append(r[:selected], r[selected+1:]...)
}

func EnumerateInitialDeals(deck cards.Set, cardsPerPlayer int, cb func(d Deal)) {
	seen := make(map[cards.Set]struct{})
	enumerateDealsHelper(deck, cards.NewSet(), cardsPerPlayer, func(p0Deal cards.Set) {
		if _, ok := seen[p0Deal]; ok {
			return
		}

		seen[p0Deal] = struct{}{}
		EnumerateDealsWithP0Hand(deck, p0Deal, cb)
	})
}

func EnumerateDealsWithP0Hand(deck, p0Deal cards.Set, cb func(d Deal)) {
	remaining := deck
	remaining.RemoveAll(p0Deal)

	seen := make(map[cards.Set]struct{})
	enumerateDealsHelper(remaining, cards.NewSet(), p0Deal.Len(), func(p1Deal cards.Set) {
		if _, ok := seen[p1Deal]; ok {
			return
		}

		seen[p1Deal] = struct{}{}
		drawPile := remaining
		drawPile.RemoveAll(p1Deal)
		drawPile.Add(cards.Defuse)
		drawPile.Add(cards.ExplodingKitten)
		seenShuffles := make(map[cards.Stack]struct{})
		EnumerateShuffles(drawPile, func(shuffle cards.Stack) {
			if _, ok := seenShuffles[shuffle]; ok {
				return
			}

			seenShuffles[shuffle] = struct{}{}
			deal := Deal{
				DrawPile: shuffle,
				P0Deal:   p0Deal,
				P1Deal:   p1Deal,
			}

			deal.P0Deal.Add(cards.Defuse)
			deal.P1Deal.Add(cards.Defuse)
			cb(deal)
		})
	})
}

func EnumerateDealsWithP1Hand(deck, p1Hand cards.Set, cb func(d Deal)) {
	EnumerateDealsWithP0Hand(deck, p1Hand, func(d Deal) {
		d.P0Deal, d.P1Deal = d.P1Deal, d.P0Deal
		cb(d)
	})
}

func enumerateDealsHelper(deck cards.Set, result cards.Set, n int, cb func(deal cards.Set)) {
	if n == 0 {
		cb(result)
		return
	}

	deck.Iter(func(card cards.Card, count uint8) {
		remaining := deck
		remaining.Remove(card)
		newResult := result
		newResult.Add(card)
		enumerateDealsHelper(remaining, newResult, n-1, cb)
	})
}

func CountDistinctShuffles(deck cards.Set) int {
	result := factorial[deck.Len()]
	deck.Iter(func(card cards.Card, count uint8) {
		result /= factorial[int(count)]
	})
	return result
}

func EnumerateShuffles(deck cards.Set, cb func(shuffle cards.Stack)) {
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
		radix := factorial[l-i-1]
		k := int(n / radix)
		n %= radix

		result.SetNthCard(i, deck.NthCard(k))
		deck.RemoveCard(k)
	}

	result.SetNthCard(l-1, deck.NthCard(0))
	return result
}

// Precomputed factorials up to 20.
var factorial = func() [20]int {
	result := [20]int{}
	result[0] = 1
	result[1] = 1
	for i := 2; i < len(result); i++ {
		result[i] = i * result[i-1]
	}
	return result
}()
