package model

import (
	"sync"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model/internal/tffloats"
)

const (
	// The number of features each history Action is encoded into.
	// This is used to size the input dimension of the network.
	numActionFeatures = 59
)

var historyPool = sync.Pool{
	New: func() interface{} {
		result := make([][]float32, gamestate.MaxNumActions)
		for i := range result {
			result[i] = make([]float32, numActionFeatures)
		}
		return result
	},
}

var actionPool = sync.Pool{
	New: func() interface{} {
		return make([]float32, numActionFeatures)
	},
}

var handPool = sync.Pool{
	New: func() interface{} {
		return make([]float32, cards.NumTypes)
	},
}

func encodeHistoryTF(h gamestate.History, result []byte) {
	oneHot := historyPool.Get().([][]float32)
	EncodeHistory(h, oneHot)
	for i, row := range oneHot {
		resultOffset := result[4*i*len(row) : 4*(i+1)*len(row)]
		tffloats.EncodeF32s(row, resultOffset)
	}
	historyPool.Put(oneHot)
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
	oneHot := actionPool.Get().([]float32)
	encodeAction(action, oneHot)
	tffloats.EncodeF32s(oneHot, result)
	actionPool.Put(oneHot)
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
	oneHot := handPool.Get().([]float32)
	encodeHand(hand, oneHot)
	tffloats.EncodeF32s(oneHot, result)
	handPool.Put(oneHot)
}

func encodeHand(hand cards.Set, result []float32) {
	clear(result)
	for _, card := range hand.AsSlice() {
		result[int(card)] += 1.0
	}
}

func clear(result []float32) {
	for i := range result {
		result[i] = 0
	}
}
