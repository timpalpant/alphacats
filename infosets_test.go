package alphacats

import (
	"testing"

	"github.com/timpalpant/alphacats/cards"
)

func TestNewInfoSetFromInitialDeal(t *testing.T) {
	deal := cards.NewSet([]cards.Card{cards.Skip, cards.Skip, cards.SeeTheFuture, cards.Shuffle})
	is := NewInfoSetFromInitialDeal(deal)

	if err := is.Validate(); err != nil {
		t.Errorf("new InfoSet is invalid: %v", err)
	}

	// Our hand should have 5 cards (including the Defuse).
	if is.OurHand.Len() != 5 {
		t.Errorf("our hand has %d cards, expected %d", is.OurHand.Len(), 5)
	}

	// Opponent hand should have 5 cards (1 known Defuse).
	if is.OpponentHand.Len() != 5 {
		t.Errorf("opponent hand has %d cards, expected %d", is.OpponentHand.Len(), 5)
	}
	if is.OpponentHand.CountOf(cards.Unknown) != 4 {
		t.Errorf("opponent hand has %d Unknown cards, expected %d",
			is.OpponentHand.CountOf(cards.Unknown), 4)
	}
	if is.OpponentHand.CountOf(cards.Defuse) != 1 {
		t.Errorf("opponent hand has %d Defuse cards, expected %d",
			is.OpponentHand.CountOf(cards.Defuse), 1)
	}

	// Draw pile should have 13 cards.
	if is.DrawPile.Len() != 13 {
		t.Errorf("draw pile has %d cards, expected %d", is.DrawPile.Len(), 13)
	}

	// There are 16 outstanding unknown cards,
	// 12 in the draw pile and 4 in our opponent's hand.
	if is.RemainingCards.Len() != 12+4 {
		t.Errorf("%d remaining Unknown cards, expected %d", is.RemainingCards.Len(), 12+4)
	}
}

func TestDrawCard_Unknown(t *testing.T) {
	t.Fail()
}

func TestDrawCard_Known(t *testing.T) {
	t.Fail()
}

func TestPlayCard(t *testing.T) {
	t.Fail()
}

func TestOpponentDrewCard_Known(t *testing.T) {
	t.Fail()
}

func TestOpponentDrewCard_Unknown(t *testing.T) {
	t.Fail()
}

func TestOpponentPlayedCard_Known(t *testing.T) {
	t.Fail()
}

func TestOpponentPlayedCard_Unknown(t *testing.T) {
	t.Fail()
}
