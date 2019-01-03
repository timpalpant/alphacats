package alphacats

// InfoSet defines a minimal and hashable representation of our current
// state of knowledge within a game, including that of:
//   1) The cards in our hand,
//   2) Any cards we know to be in the other player's hand,
//   3) The cards in the draw pile, and their positions if we know it,
//   4) The outstanding set of cards that must be in play somewhere,
//      but whose location we do not yet know.
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

func NewInfoSetFromInitialDeal(deal CardSet) InfoSet {
	ourHand := deal
	ourHand[Defuse] += 1

	opponentHand := CardSet{}
	opponentHand[Defuse] = 1
	opponentHand[Unknown] = 4

	drawPile := CardSet{}
	drawPile[ExplodingCat] = 1
	drawPile[Unknown] = 12

	remainingCards := CoreDeck.Remove(deal)

	return InfoSet{
		OurHand:        ourHand,
		OpponentHand:   opponentHand,
		DrawPile:       drawPile,
		RemainingCards: remainingCards,
	}
}

func EnumerateInitialInfoSets() []InfoSet {
	deals := enumerateInitialDeals(CoreDeck, CardSet{}, Unknown, 4, nil)
	result := make([]InfoSet, len(deals))
	for i, deal := range deals {
		result[i] = NewInfoSetFromInitialDeal(deal)
	}

	return result
}

func enumerateInitialDeals(available CardSet, current CardSet, start Card, desired int, result []CardSet) []CardSet {
	nRemaining := uint8(desired - current.Count())
	if nRemaining == 0 {
		return append(result, current)
	}

	for card := start; card <= Cat; card++ {
		count := available[card]
		for i := uint8(0); i <= min(count, nRemaining); i++ {
			current[card] += i
			available[card] -= i
			result = enumerateInitialDeals(available, current, card+1, desired, result)
			current[card] -= i
			available[card] += i
		}
	}

	return result
}

func min(i, j uint8) uint8 {
	if i < j {
		return i
	}

	return j
}

type Strategy struct {
	// For each info set => available actions => probability we take it.
	actions map[InfoSet]map[Action]float64
}
