package gamestate

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
)

// PrivateInfo holds the private hidden information that either player has.
type privateInfo struct {
	// The Cards we have in our hand. All Cards should be known.
	ourHand cards.Set
	// The Cards our opponent has in their hand. Some Cards may be Unknown.
	opponentHand cards.Set
	// Cards that we know in the draw pile. For example, after playing a
	// SeeTheFuture card we know the identity of the top three cards.
	knownDrawPileCards cards.Stack
}

// Return a new PrivateInfo created as if the player is dealt the given
// Set of (4) cards at the beginning of the game, not including the
// Defuse card that is always added.
func newPrivateInfo(deal cards.Set) privateInfo {
	if deal.Len() != 4 {
		panic(fmt.Errorf("initial deal must have 4 cards, got %d", deal.Len()))
	}

	ourHand := deal
	ourHand.Add(cards.Defuse)

	opponentHand := cards.NewSet()
	opponentHand.Add(cards.Defuse)
	for i := 0; i < 4; i++ {
		opponentHand.Add(cards.Unknown)
	}

	return privateInfo{
		ourHand:      ourHand,
		opponentHand: opponentHand,
	}
}

// Verifies that the InfoSet is valid and satisifes all internal constraints.
func (pi *privateInfo) validate() error {
	// All cards in our hand must be known.
	if pi.ourHand.CountOf(cards.Unknown) != 0 {
		return fmt.Errorf("found Unknown cards in our hand")
	}

	return nil
}

// Modify InfoSet as if we drew the given Card from the top of the draw pile.
func (pi *privateInfo) drawCard(card cards.Card, fromPosition int) {
	known := pi.knownDrawPileCards.NthCard(fromPosition)
	if known != cards.Unknown && known != card {
		panic(fmt.Errorf("drew card %v from position %v but we knew it to be %v",
			card, fromPosition, known))
	}

	// Add card to our hand.
	// NOTE: Must be after above calculation of # cards in draw pile,
	// since that is backed out from the number of cards in our hand.
	pi.ourHand.Add(card)
	pi.knownDrawPileCards.RemoveCard(fromPosition)
}

func (pi *privateInfo) playCard(card cards.Card) {
	pi.ourHand.Remove(card)
}

// Modify PrivateInfo to reflect our opponent drawing a card from
// the draw pile.
func (pi *privateInfo) opponentDrewCard(card cards.Card, fromPosition int) {
	// If we knew what the card in the pile was, we now know it is in their hand.
	// Otherwise an Unknown card is added to their hand.
	drawnCard := pi.knownDrawPileCards.NthCard(fromPosition)
	if drawnCard != cards.Unknown && drawnCard != card {
		panic(fmt.Errorf("drew card %v from position %v but we knew it to be %v",
			card, fromPosition, drawnCard))
	}

	// If they drew an exploding cat then we see it either way.
	if card == cards.ExplodingCat {
		drawnCard = cards.ExplodingCat
	}

	pi.knownDrawPileCards.RemoveCard(fromPosition)
	pi.opponentHand.Add(drawnCard)
}

// OpponentPlayedCard modifies the PrivateInfo to reflect the fact
// that our opponent played the given card.
func (pi *privateInfo) opponentPlayedCard(card cards.Card) {
	if pi.opponentHand.CountOf(card) > 0 {
		// We knew the player had this card.
		pi.opponentHand.Remove(card)
	} else {
		// We didn't know they had this card; it was one of the Unknown's
		// in their hand.
		pi.opponentHand.Remove(cards.Unknown)
	}
}

// Modify PrivateInfo to reflect seeing these cards on the top
// of the draw pile.
func (pi *privateInfo) seeTopCards(topN []cards.Card) {
	for i, card := range topN {
		nthCard := pi.knownDrawPileCards.NthCard(i)
		if nthCard != cards.Unknown && nthCard != card {
			panic(fmt.Errorf("we knew %d th card to be %v, but are now told it is %v",
				i, nthCard, card))
		}

		pi.knownDrawPileCards.SetNthCard(i, card)
	}
}
