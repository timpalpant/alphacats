package alphacats

import (
	"github.com/timpalpant/alphacats/cards"
)

func EnumerateGames(cb func(drawPile cards.Stack, p0Deal, p1Deal cards.Set)) {
	// Deal 4 cards to player 0.
	player0Deals := enumerateInitialDeals(cards.CoreDeck, cards.NewSet(), cards.Unknown, 4, nil)
	for _, p0Deal := range player0Deals {
		remainingCards := cards.CoreDeck
		remainingCards.RemoveAll(p0Deal)
		// Deal 4 cards to player 1.
		player1Deals := enumerateInitialDeals(remainingCards, cards.NewSet(), cards.Unknown, 4, nil)
		for _, p1Deal := range player1Deals {
			drawPile := cards.CoreDeck
			drawPile.RemoveAll(p0Deal)
			drawPile.RemoveAll(p1Deal)
			drawPile.Add(cards.ExplodingCat)
			p0 := p0Deal
			p0.Add(cards.Defuse)
			p1 := p1Deal
			p1.Add(cards.Defuse)

			enumerateShuffles(drawPile, func(shuffle cards.Stack) {
				cb(shuffle, p0Deal, p1Deal)
			})
		}
	}
}

func enumerateInitialDeals(available cards.Set, current cards.Set, card cards.Card, desired int, result []cards.Set) []cards.Set {
	if card > cards.Cat {
		return result
	}

	nRemaining := desired - current.Len()
	if nRemaining == 0 || nRemaining > available.Len() {
		return append(result, current)
	}

	count := int(available.CountOf(card))
	for i := 0; i <= min(count, nRemaining); i++ {
		current.AddN(card, i)
		available.RemoveN(card, i)
		result = enumerateInitialDeals(available, current, card+1, desired, result)
		current.RemoveN(card, i)
		available.AddN(card, i)
	}

	return result
}
