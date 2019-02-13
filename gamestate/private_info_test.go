package gamestate

import (
	"testing"

	"github.com/timpalpant/alphacats/cards"
)

func TestNewPrivateInfo(t *testing.T) {
	testCards := []cards.Card{cards.Skip, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))

	if pi.ourHand.Len() != 5 {
		t.Errorf("initial hand should have 5 cards, got %v", pi.ourHand.Len())
	}
	if !pi.ourHand.Contains(cards.Defuse) {
		t.Errorf("initial hand should have a Defuse card")
	}
	for _, card := range testCards {
		if !pi.ourHand.Contains(card) {
			t.Errorf("initial hand should have a %v card", card)
		}
	}

	if pi.opponentHand.Len() != 5 {
		t.Errorf("initial hand should have 5 cards, got %v", pi.ourHand.Len())
	}
	if !pi.opponentHand.Contains(cards.Defuse) {
		t.Errorf("initial hand should have a Defuse card")
	}
}

func TestDrawCard(t *testing.T) {
	testCards := []cards.Card{cards.Skip, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))
	pi.knownDrawPileCards.SetNthCard(1, cards.ExplodingCat)
	pi.drawCard(cards.Slap2x, 0)

	if pi.ourHand.CountOf(cards.Slap2x) != 1 {
		t.Error("our hand should contain drawn card")
	}

	if pi.knownDrawPileCards.NthCard(0) != cards.ExplodingCat {
		t.Error("known draw pile cards should be shifted by one")
	}
}

func TestOpponentDrewCard_Unknown(t *testing.T) {
	testCards := []cards.Card{cards.Skip, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))
	pi.knownDrawPileCards.SetNthCard(1, cards.ExplodingCat)
	pi.opponentDrewCard(cards.Slap2x, 0)

	if pi.opponentHand.Len() != 6 {
		t.Error("opponent should have card added to their hand")
	}

	if pi.opponentHand.CountOf(cards.Slap2x) != 0 {
		t.Error("opponent's drawn card should be kept secret")
	}

	if pi.opponentHand.CountOf(cards.Unknown) != 5 {
		t.Error("number of unknown cards in opponent's hand should increase by one")
	}
}

func TestOpponentDrewCard_Known(t *testing.T) {
	testCards := []cards.Card{cards.Skip, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))
	pi.knownDrawPileCards.SetNthCard(0, cards.Slap2x)
	pi.opponentDrewCard(cards.Slap2x, 0)

	if pi.opponentHand.Len() != 6 {
		t.Error("opponent should have card added to their hand")
	}

	if pi.opponentHand.CountOf(cards.Slap2x) != 1 {
		t.Error("opponent's drawn card was known")
	}

	if pi.opponentHand.CountOf(cards.Unknown) != 4 {
		t.Error("number of unknown cards in opponent's hand should not change")
	}
}

func TestPlayCard(t *testing.T) {
	testCards := []cards.Card{cards.Skip, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))
	pi.knownDrawPileCards.SetNthCard(1, cards.ExplodingCat)

	pi.playCard(cards.Slap1x)

	if pi.ourHand.Len() != 4 {
		t.Error("one card should be removed from our hand")
	}

	if pi.ourHand.Contains(cards.Slap1x) {
		t.Error("Slap1x card should be removed from our hand")
	}

	if pi.knownDrawPileCards.IsEmpty() {
		t.Error("known draw pile cards should not be reset")
	}
}

func TestPlayCard_Shuffle(t *testing.T) {
	testCards := []cards.Card{cards.Shuffle, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))
	pi.knownDrawPileCards.SetNthCard(1, cards.ExplodingCat)

	pi.playCard(cards.Shuffle)

	if pi.ourHand.Len() != 4 {
		t.Error("one card should be removed from our hand")
	}

	if pi.ourHand.Contains(cards.Shuffle) {
		t.Error("Shuffle card should be removed from our hand")
	}

	if !pi.knownDrawPileCards.IsEmpty() {
		t.Error("known draw pile cards should be reset after playing shuffle")
	}
}

func TestOpponentPlayedCard_Unknown(t *testing.T) {
	testCards := []cards.Card{cards.Shuffle, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))

	pi.opponentPlayedCard(cards.Shuffle)

	if pi.opponentHand.Len() != 4 {
		t.Error("one card should be removed from opponent's hand")
	}

	if pi.opponentHand.CountOf(cards.Unknown) != 3 {
		t.Error("one Unknown card should be removed from opponent's hand")
	}
}

func TestOpponentPlayedCard_Known(t *testing.T) {
	testCards := []cards.Card{cards.Shuffle, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))
	pi.knownDrawPileCards.SetNthCard(1, cards.ExplodingCat)

	pi.opponentPlayedCard(cards.Defuse)

	if pi.opponentHand.Len() != 4 {
		t.Error("one card should be removed from opponent's hand")
	}

	if pi.opponentHand.CountOf(cards.Unknown) != 4 {
		t.Error("known Defuse card should be removed from opponent's hand")
	}

	if pi.opponentHand.CountOf(cards.Defuse) != 0 {
		t.Error("known Defuse card should be removed from opponent's hand")
	}

	if pi.knownDrawPileCards.IsEmpty() {
		t.Error("known draw pile cards should not be reset after playing Defuse")
	}
}

func TestOpponentPlayedCard_Shuffle(t *testing.T) {
	testCards := []cards.Card{cards.Shuffle, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))
	pi.knownDrawPileCards.SetNthCard(1, cards.ExplodingCat)

	pi.opponentPlayedCard(cards.Shuffle)

	if !pi.knownDrawPileCards.IsEmpty() {
		t.Error("known draw pile cards should be reset after playing shuffle")
	}
}

func TestSeeTopCards(t *testing.T) {
	testCards := []cards.Card{cards.Shuffle, cards.Cat, cards.Slap1x, cards.SeeTheFuture}
	pi := newPrivateInfo(cards.NewSetFromCards(testCards))

	topCards := []cards.Card{cards.ExplodingCat, cards.Cat, cards.Skip}
	pi.seeTopCards(topCards)

	for i, card := range topCards {
		if pi.knownDrawPileCards.NthCard(i) != card {
			t.Errorf("player should know %d th card is %v", i, card)
		}
	}
}
