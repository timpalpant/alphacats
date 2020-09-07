package model

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model/internal/tffloats"
)

var (
	deck = append(cards.CoreDeck.AsSlice(),
		cards.ExplodingKitten,
		cards.Defuse,
		cards.Defuse,
		cards.Defuse)
)

const (
	// The number of features each history Action is encoded into.
	// This is used to size the input dimension of the network.
	numActionFeatures  = 16
	numCardsInDeck = 23
	maxCardsInDrawPile = 13
	// Vector size of output predictions: one for each card type,
	// one for each insertion position, and one for drawing a card.
	// One for each type of card to play or give.
	maxInsertPositions = 8
	outputDimension    = 2*cards.NumTypes + maxInsertPositions + 1
)

func encodeHistoryTF(h gamestate.History, result []byte) {
	// We encode actions directly, rather than reuse EncodeHistory,
	// to avoid needing to allocate large intermediate one-hot [][]float32.
	idx := 0
	for i := 0; i < h.Len(); i++ {
		encodeActionTF(h.Get(i), result[idx:])
		idx += 4 * numActionFeatures
	}

	for i := idx; i < len(result); i++ {
		result[i] = 0
	}
}

func newOneHotHistory() [][]float32 {
	result := make([][]float32, gamestate.MaxNumActions)
	for i := range result {
		result[i] = make([]float32, numActionFeatures)
	}
	return result
}

// Game history is encoded as: MaxNumActions (58) x
//  - One hot encoded player (2)
//  - One hot encoded action type (4)
//  - One hot encoded Card (10)
//  - One hot encoded position in draw pile (13)
//  - Concatenated one hot cards seen (3x10)
func EncodeHistory(h gamestate.History, result [][]float32) {
	for i := 0; i < h.Len(); i++ {
		encodeAction(h.Get(i), result[i])
	}

	for i := h.Len(); i < len(result); i++ {
		clear(result[i])
	}
}

func encodeActionTF(action gamestate.Action, result []byte) {
	var oneHot [numActionFeatures]float32
	encodeAction(action, oneHot[:])
	tffloats.EncodeF32s(oneHot[:], result)
}

func encodeAction(action gamestate.Action, result []float32) {
	clear(result)
	result[int(action.Player)] = 1.0
	result[2+int(action.Type)-1] = 1.0
	result[6+int(action.Card)] = 1.0
	// NOTE: Private action info is not included in the encoding.
	// This is okay because our abstracted info set factors out private information
	// about the draw pile and our hand separately.
	//result[16+action.PositionInDrawPile] = 1.0
	//for j, card := range action.CardsSeen {
	//	if card != cards.Unknown {
	//		result[29+10*j+int(card)] = 1.0
	//	}
	//}
}

func encodeHandTF(hand cards.Set, result []byte) {
	var oneHot [numCardsInDeck]float32
	encodeHand(hand, oneHot[:])
	tffloats.EncodeF32s(oneHot[:], result)
}

func encodeHand(hand cards.Set, result []float32) {
	clear(result)
	for i, card := range deck {
		if hand.Contains(card) {
			result[i] = 1.0
			hand.Remove(card)
		}
	}

	if hand.Len() != 0 {
		panic(fmt.Errorf("cards still remain in hand!"))
	}
}

func newOneHotDrawPile() [][]float32 {
	result := make([][]float32, maxCardsInDrawPile)
	for i := range result {
		result[i] = make([]float32, cards.NumTypes)
	}
	return result
}

func encodeDrawPileTF(drawPile cards.Stack, result []byte) {
	// We encode actions directly, rather than reuse EncodeDrwawPile,
	// to avoid needing to allocate large intermediate one-hot [][]float32.
	i := 0
	drawPile.Iter(func(card cards.Card) {
		encodeCardTF(drawPile.NthCard(i), result[4*cards.NumTypes*i:])
		i++
	})

	for idx := 4 * cards.NumTypes * i; idx < len(result); idx++ {
		result[idx] = 0
	}
}

func encodeDrawPile(drawPile cards.Stack, result [][]float32) {
	i := 0
	drawPile.Iter(func(card cards.Card) {
		encodeCard(card, result[i])
		i++
	})

	for ; i < len(result); i++ {
		clear(result[i])
	}
}

func encodeCardTF(card cards.Card, result []byte) {
	var oneHot [cards.NumTypes]float32
	encodeCard(card, oneHot[:])
	tffloats.EncodeF32s(oneHot[:], result)
}

func encodeCard(card cards.Card, result []float32) {
	clear(result)
	result[card] = 1.0
}

func encodeOutputMaskTF(numDrawPileCards int, availableActions []gamestate.Action, result []byte) {
	var mask [outputDimension]float32
	encodeOutputMask(numDrawPileCards, availableActions, mask[:])
	tffloats.EncodeF32s(mask[:], result)
}

func encodeOutputMask(numDrawPileCards int, availableActions []gamestate.Action, result []float32) {
	clear(result)
	for _, action := range availableActions {
		switch action.Type {
		case gamestate.DrawCard:
			// First position is always the advantages of ending turn by drawing a card,
			// since this corresponds to the "Unknown" card enum.
			result[0] = 1.0
		case gamestate.PlayCard:
			// Next 10 positions correspond to playing each card type.
			result[action.Card] = 1.0
		case gamestate.GiveCard:
			// Next 10 positions correspond to giving each card type.
			result[cards.NumTypes+int(action.Card)] = 1.0
		case gamestate.InsertExplodingKitten:
			// Remaining correspond to inserting cat at each position.
			// Position 0 -> insert on the bottom.
			// Position 1 -> insert randomly.
			// Position 2...N -> insert in the nth position.
			idx := 2*cards.NumTypes + 1 + int(action.PositionInDrawPile)
			if int(action.PositionInDrawPile) == numDrawPileCards+1 {
				idx = 2*cards.NumTypes
			}

			result[idx] = 1.0
		default:
			panic(fmt.Errorf("unsupported action: %v", action))
		}
	}
}

func encodeOutputs(numDrawPileCards int, availableActions []gamestate.Action, policy, result []float32) {
	clear(result)
	for i, action := range availableActions {
		switch action.Type {
		case gamestate.DrawCard:
			// First position is always the advantages of ending turn by drawing a card,
			// since this corresponds to the "Unknown" card enum.
			result[0] = policy[i]
		case gamestate.PlayCard:
			// Next 10 positions correspond to playing each card type.
			result[action.Card] = policy[i]
		case gamestate.GiveCard:
			// Next 10 positions correspond to giving each card type.
			result[cards.NumTypes+int(action.Card)] = policy[i]
		case gamestate.InsertExplodingKitten:
			// Remaining correspond to inserting cat at each position.
			// Position 0 -> insert on the bottom.
			// Position 1 -> insert randomly.
			// Position 2...N -> insert in the nth position.
			idx := 2*cards.NumTypes + 1 + int(action.PositionInDrawPile)
			if int(action.PositionInDrawPile) == numDrawPileCards+1 {
				idx = 2*cards.NumTypes
			}

			result[idx] = policy[i]
		default:
			panic(fmt.Errorf("unsupported action: %v", action))
		}
	}
}

func decodeOutputs(numDrawPileCards int, availableActions []gamestate.Action, predictions []float32) []float32 {
	policy := make([]float32, len(availableActions))
	for i, action := range availableActions {
		switch action.Type {
		case gamestate.DrawCard:
			// First position is always the advantages of ending turn by drawing a card,
			// since this corresponds to the "Unknown" card enum.
			policy[i] = predictions[0]
		case gamestate.PlayCard:
			// Next 9 positions correspond to playing/giving each card type.
			policy[i] = predictions[action.Card]
		case gamestate.GiveCard:
			policy[i] = predictions[cards.NumTypes+int(action.Card)]
		case gamestate.InsertExplodingKitten:
			// Remaining correspond to inserting cat at each position.
			idx := 2*cards.NumTypes + 1 + int(action.PositionInDrawPile)
			if int(action.PositionInDrawPile) == numDrawPileCards+1 {
				idx = 2*cards.NumTypes
			}
			policy[i] = predictions[idx]
		default:
			panic(fmt.Errorf("unsupported action: %v", action))
		}
	}

	// Renormalize policy since some weight may have been given to invalid actions.
	normalize(policy)
	return policy
}

func normalize(p []float32) {
	total := sum(p)
	for i := range p {
		p[i] /= total
	}
}

func sum(vs []float32) float32 {
	total := float32(0.0)
	for _, v := range vs {
		total += v
	}
	return total
}

func clear(result []float32) {
	for i := range result {
		result[i] = 0
	}
}
