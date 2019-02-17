package gamestate

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
)

type DrawPile struct {
	// Set of Cards remaining in the draw pile.
	remaining cards.Set
	// Cards in the draw pile whose identity is fixed because one of the player's
	// knows it.
	fixed cards.Stack
}

func NewDrawPile(remaining cards.Set) DrawPile {
	return DrawPile{remaining: remaining}
}

func (dp DrawPile) String() string {
	return fmt.Sprintf("{remaining: %s, fixed: %s}", dp.remaining, dp.fixed)
}

func (dp DrawPile) Len() int {
	return dp.remaining.Len()
}

func (dp *DrawPile) drawCard(card cards.Card) {
	dp.remaining.Remove(card)
	dp.fixed.RemoveCard(0)
}

func (dp *DrawPile) drawCardFromBottom(card cards.Card) {
	bottom := dp.remaining.Len() - 1
	dp.remaining.Remove(card)
	dp.fixed.RemoveCard(bottom)
}

func (dp *DrawPile) insert(card cards.Card, position int) {
	dp.remaining.Add(cards.ExplodingCat)
	dp.fixed.InsertCard(cards.ExplodingCat, position)
}

func (dp *DrawPile) fixTop3Cards(top3 [3]cards.Card) {
	for i, card := range top3 {
		nthCard := dp.fixed.NthCard(i)
		if nthCard != cards.Unknown && nthCard != card {
			panic(fmt.Errorf("we knew %d th card to be %v, but are now told it is %v",
				i, nthCard, card))
		}

		dp.fixed.SetNthCard(i, card)
	}
}

func (dp DrawPile) BottomCardProbabilities() map[cards.Card]float64 {
	bottom := dp.remaining.Len() - 1
	bottomCard := dp.fixed.NthCard(bottom)
	if bottomCard != cards.Unknown {
		// Identity of the bottom card is fixed.
		return fixedCardProbabilities[bottomCard]
	}

	// Note: We need to exclude any cards whose identity is already fixed in a
	// position known *not* to be the bottom card.
	candidates := dp.remaining
	for i := 0; i < bottom; i++ {
		if card := dp.fixed.NthCard(i); card != cards.Unknown {
			candidates.Remove(card)
		}
	}

	result, ok := cardProbabilitiesCache[candidates]
	if !ok {
		panic(fmt.Errorf("missing card probabilities for: %v", candidates))
	}

	return result
}

func (dp DrawPile) TopCardProbabilities() map[cards.Card]float64 {
	topCard := dp.fixed.NthCard(0)
	if topCard != cards.Unknown {
		return fixedCardProbabilities[topCard]
	}

	// Note: We need to exclude any cards whose identity is already fixed in a
	// position known *not* to be the top card.
	start := 1
	end := dp.remaining.Len()
	candidates := dp.remaining
	for i := start; i < end; i++ {
		if card := dp.fixed.NthCard(i); card != cards.Unknown {
			candidates.Remove(card)
		}
	}

	result, ok := cardProbabilitiesCache[candidates]
	if !ok {
		panic(fmt.Errorf("missing card probabilities for: %v", candidates))
	}

	return result
}
