package alphacats

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
)

// InfoSet defines a minimal and hashable representation of a Player's
// current state of knowledge within a game, including that of:
//   1) The cards in our hand,
//   2) Any cards we know to be in the other player's hand,
//   3) The cards that have already been played (discard pile),
//   4) Fixed cards in the draw pile, if we know their position
//
// Because an InfoSet completely defines a player's point of view
// within the game, a Player strategy must have a single action policy
// for each distinct InfoSet.
type InfoSet struct {
	// The Cards we have in our hand. All Cards should be known.
	OurHand cards.Set
	// The Cards our opponent has in their hand. Some Cards may be Unknown.
	OpponentHand cards.Set
	// The Cards that have been played in the discard pile.
	// All Cards should be known.
	DiscardPile cards.Set
	// Cards that we know in the draw pile. For example, after playing a
	// SeeTheFuture card we know the identity of the top three cards.
	KnownDrawPileCards cards.Stack
}

// Verifies that the InfoSet is valid and satisifes all internal constraints.
func (is InfoSet) Validate() error {
	// All cards in our hand must be known.
	if is.OurHand.CountOf(cards.Unknown) != 0 {
		return fmt.Errorf("found Unknown cards in our hand")
	}

	if is.DiscardPile.CountOf(cards.Unknown) != 0 {
		return fmt.Errorf("found Unknown cards in the discard pile")
	}

	return nil
}

func (is InfoSet) NumCardsInDrawPile() int {
	nCardsTotal := cards.CoreDeck.Len() + 3 // ExplodingCat + Defuse for each player.
	return nCardsTotal - is.OurHand.Len() - is.OpponentHand.Len() - is.DiscardPile.Len()
}

// Return a new InfoSet created as if the player is dealt the given
// Set of (4) cards at the beginning of the game, not including the
// Defuse card that is always added.
func NewInfoSetFromInitialDeal(deal cards.Set) InfoSet {
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

	return InfoSet{
		OurHand:      ourHand,
		OpponentHand: opponentHand,
	}
}

// Modify InfoSet as if we drew the given Card from the top of the draw pile.
func (is *InfoSet) DrawCard(card cards.Card, fromBottom bool) {
	// Shift our known draw pile cards up by one.
	position := 0
	if fromBottom {
		position = is.NumCardsInDrawPile() - 1
	}

	known := is.KnownDrawPileCards.NthCard(position)
	if known != cards.Unknown && known != card {
		panic(fmt.Errorf("drew card %v from %v (from bottom: %v) but we knew it to be %v",
			card, position, fromBottom, known))
	}

	// Add card to our hand.
	// NOTE: Must be after above calculation of # cards in draw pile,
	// since that is backed out from the number of cards in our hand.
	is.OurHand.Add(card)
	is.KnownDrawPileCards.RemoveCard(position)

	if err := is.Validate(); err != nil {
		panic(err)
	}
}

func (is *InfoSet) PlayCard(card cards.Card) {
	is.OurHand.Remove(card)
	is.DiscardPile.Add(card)

	if err := is.Validate(); err != nil {
		panic(err)
	}
}

// Modify InfoSet to reflect our opponent drawing the top card
// of the draw pile.
func (is *InfoSet) OpponentDrewCard(card cards.Card, fromBottom bool) {
	position := 0
	if fromBottom {
		position = is.NumCardsInDrawPile() - 1
	}

	// If we knew what the card in the pile was, we now know it is in their hand.
	// Otherwise an Unknown card is added to their hand.
	drawnCard := is.KnownDrawPileCards.NthCard(position)
	if drawnCard != cards.Unknown && drawnCard != card {
		panic(fmt.Errorf("drew card %v from %v (from bottom: %v) but we knew it to be %v",
			card, position, fromBottom, drawnCard))
	}

	// If they drew an exploding cat then we see it either way.
	if card == cards.ExplodingCat {
		drawnCard = cards.ExplodingCat
	}

	is.KnownDrawPileCards.RemoveCard(position)
	is.OpponentHand.Add(drawnCard)

	if err := is.Validate(); err != nil {
		panic(err)
	}
}

// OpponentPlayedCard modifies the infoset to reflect the fact
// that our opponent played the given card.
func (is *InfoSet) OpponentPlayedCard(card cards.Card) {
	is.DiscardPile.Add(card)

	if is.OpponentHand.CountOf(card) > 0 {
		// We knew the player had this card.
		is.OpponentHand.Remove(card)
	} else {
		// We didn't know they had this card; it was one of the Unknown's
		// in their hand.
		is.OpponentHand.Remove(cards.Unknown)
	}

	if err := is.Validate(); err != nil {
		panic(err)
	}
}

// Modify InfoSet to reflect seeing these cards on the top
// of the draw pile.
func (is *InfoSet) SeeTopCards(topN []cards.Card) {
	for i, card := range topN {
		nthCard := is.KnownDrawPileCards.NthCard(i)
		if nthCard != cards.Unknown && nthCard != card {
			panic(fmt.Errorf("we knew %d th card to be %v, but are now told it is %v",
				i, nthCard, card))
		}

		is.KnownDrawPileCards.SetNthCard(i, card)
	}

	if err := is.Validate(); err != nil {
		panic(err)
	}
}
