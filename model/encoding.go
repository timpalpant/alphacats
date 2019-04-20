package model

import (
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model/internal/tffloats"
)

const (
	// The number of features each history Action is encoded into.
	// This is used to size the input dimension of the network.
	numActionFeatures = 59
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

// Game history is encoded as: MaxHistory (48) x
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
	result[16+action.PositionInDrawPile] = 1.0
	for j, card := range action.CardsSeen {
		if card != cards.Unknown {
			result[29+10*j+int(card)] = 1.0
		}
	}
}

func encodeHandTF(hand cards.Set, result []byte) {
	var oneHot [cards.NumTypes]float32
	encodeHand(hand, oneHot[:])
	tffloats.EncodeF32s(oneHot[:], result)
}

func encodeHand(hand cards.Set, result []float32) {
	clear(result)
	hand.Iter(func(card cards.Card, count uint8) {
		result[int(card)] = float32(count)
	})
}

func clear(result []float32) {
	for i := range result {
		result[i] = 0
	}
}
