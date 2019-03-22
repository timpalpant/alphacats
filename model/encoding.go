package model

import (
	"fmt"

	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

const (
	// The maximum number of choices a player ever has at any game node.
	// This is used to size the output dimension of the network.
	maxNumChoices = 16
	// The number of features each history Action is encoded into.
	// This is used to size the input dimension of the network.
	numActionFeatures = 59
)

func encodeHistories(samples []deepcfr.Sample) []float32 {
	n := len(samples) * gamestate.MaxNumActions * numActionFeatures
	result := make([]float32, 0, n)
	for _, sample := range samples {
		is := sample.InfoSet.(*alphacats.InfoSetWithAvailableActions)
		X := EncodeHistory(is.History)
		for _, row := range X {
			result = append(result, row...)
		}
	}

	return result
}

// Game history is encoded as: MaxHistory (48) x
//  - One hot encoded player (2)
//  - One hot encoded action type (4)
//  - One hot encoded Card (10)
//  - One hot encoded position in draw pile (13)
//  - Concatenated one hot cards seen (3x10)
func EncodeHistory(h []gamestate.EncodedAction) [][]float32 {
	result := make([][]float32, gamestate.MaxNumActions)

	for i, action := range h {
		result[i] = encodeAction(action.Decode())
	}

	for i := len(h); i < len(result); i++ {
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

func encodeHands(samples []deepcfr.Sample) []float32 {
	result := make([]float32, 0, len(samples)*cards.NumTypes)
	for _, sample := range samples {
		is := sample.InfoSet.(*alphacats.InfoSetWithAvailableActions)
		X := encodeHand(is.Hand)
		result = append(result, X...)
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

func encodeTargets(samples []deepcfr.Sample) []float32 {
	result := make([]float32, 0, len(samples)*maxNumChoices)
	for _, sample := range samples {
		y := encodeAdvantages(sample.Advantages)
		result = append(result, y...)
	}
	return result
}

// Pad advantages to make fixed-size output vector.
// NODE: We place them in the first N positions, which needs to match
// the encoding of the numActions mask below.
func encodeAdvantages(advantages []float32) []float32 {
	if len(advantages) > maxNumChoices {
		panic(fmt.Errorf("%d advantages > expected max %d",
			len(advantages), maxNumChoices))
	}

	result := make([]float32, maxNumChoices)
	copy(result, advantages)
	return result
}

func encodeSampleWeights(batch []deepcfr.Sample) []float32 {
	result := make([]float32, len(batch))
	for i, sample := range batch {
		w := float32((sample.Iter + 1) / 2)
		result[i] = w
	}

	return result
}

func encodeActions(batch []deepcfr.Sample) []float32 {
	var result []float32
	for _, sample := range batch {
		is := sample.InfoSet.(*alphacats.InfoSetWithAvailableActions)
		for _, action := range is.AvailableActions {
			result = append(result, encodeAction(action)...)
		}
	}

	return result
}
