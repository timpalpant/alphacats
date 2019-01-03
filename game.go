package alphacats

type Action int

const (
	DrawCard Action = iota
	PlayDefuseCard
	PlaySkipCard
	PlaySlap1xCard
	PlaySlap2xCard
	PlaySeeTheFutureCard
	PlayShuffleCard
	PLayDrawFromTheBottomCard
	PlayCatCard
)

type InfoSet struct {
	// The Cards we have in our hand. All Cards should be known.
	OurHand CardSet
	// The Cards our opponent has in their hand. Some Cards may be Unknown.
	OpponentHand CardSet
	// The Cards remaining in the draw pile. Some Cards may be Unknown.
	DrawPile CardSet
	// The remaining Cards whose location we do not know. These may be
	// in our opponent's hand or in the draw pile. The number of cards should
	// correspond to the total number of Unknown Cards in the OpponentHand
	// and the DrawPile.
	RemainingCards CardSet
	// Cards that we know in the draw pile. For example, after playing a
	// SeeTheFuture card we know the identity of the top three cards.
	KnownDrawPileCards CardPile
}

type Strategy struct {
	// For each info set => available actions => probability we take it.
	actions map[InfoSet]map[Action]float64
}

func Deal() InfoSet {
	deck := CoreDeck

	// We are dealt 5 cards, one of which is a Defuse.
	ourHand := deck.DrawRandom(4)
	deck = deck.Remove(ourHand)
	ourHand[Defuse] += 1

	// Opponent is dealt 5 of the cards, one of which we know is a Defuse.
	opponentHand := CardSet{}
	opponentHand[Defuse] = 1
	opponentHand[Unknown] = 4

	// Draw pile has 13 cards, one of which we know is an ExplodingCat.
	drawPile := CardSet{}
	drawPile[ExplodingCat] = 1
	drawPile[Unknown] = 12

	return InfoSet{
		OurHand:        ourHand,
		OpponentHand:   opponentHand,
		DrawPile:       drawPile,
		RemainingCards: deck,
	}
}
