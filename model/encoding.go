package model

import (
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

const (
	// The number of features each history Action is encoded into.
	// This is used to size the input dimension of the network.
	numActionFeatures = 59
)

// Game history is encoded as: MaxHistory (48) x
//  - One hot encoded player (2)
//  - One hot encoded action type (4)
//  - One hot encoded Card (10)
//  - One hot encoded position in draw pile (13)
//  - Concatenated one hot cards seen (3x10)
func EncodeHistory(h gamestate.History) [][]float32 {
	result := make([][]float32, gamestate.MaxNumActions)

	for i := 0; i < h.Len(); i++ {
		result[i] = encodeAction(h.Get(i))
	}

	for i := h.Len(); i < len(result); i++ {
		result[i] = make([]float32, numActionFeatures)
	}

	return result
}

func encodeAction(action gamestate.Action) []float32 {
	result := make([]float32, numActionFeatures)
	result[int(action.Player)] = 1.0
	result[2+int(action.Type)-1] = 1.0
	result[6+int(action.Card)] = 1.0
	result[16+action.PositionInDrawPile] = 1.0
	for j, card := range action.CardsSeen {
		if card != cards.Unknown {
			result[29+10*j+int(card)] = 1.0
		}
	}

	return result
}

func encodeHand(hand cards.Set) []float32 {
	result := make([]float32, cards.NumTypes)
	for _, card := range hand.AsSlice() {
		result[int(card)] += 1.0
	}

	return result
}
