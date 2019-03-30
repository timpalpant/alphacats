package model

import (
	"fmt"

	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

const (
	// The number of features each history Action is encoded into.
	// This is used to size the input dimension of the network.
	numActionFeatures = 59
)

func encodeHistories(samples []deepcfr.Sample) []float32 {
	n := len(samples) * gamestate.MaxNumActions * numActionFeatures
	result := make([]float32, 0, n)
	for _, sample := range samples {
		is := sample.InfoSet.(*alphacats.InfoSetWithAvailableActions)
		if len(is.AvailableActions) != len(sample.Advantages) {
			panic(fmt.Errorf("Sample has %d actions but %d advantages",
				len(is.AvailableActions), len(sample.Advantages)))
		}

		X := EncodeHistory(is.History)
		for i := 0; i < len(sample.Advantages); i++ {
			for _, row := range X {
				result = append(result, row...)
			}
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
		for i := 0; i < len(sample.Advantages); i++ {
			result = append(result, X...)
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

func encodeTargets(samples []deepcfr.Sample) []float32 {
	var result []float32
	for _, sample := range samples {
		result = append(result, sample.Advantages...)
	}
	return result
}

func encodeSampleWeights(batch []deepcfr.Sample) []float32 {
	var result []float32
	for _, sample := range batch {
		w := float32(int((sample.Weight + 1.0) / 2.0))
		for i := 0; i < len(sample.Advantages); i++ {
			result = append(result, w)
		}
	}

	return result
}

func encodeActions(batch []deepcfr.Sample) []float32 {
	var result []float32
	for _, sample := range batch {
		is := sample.InfoSet.(*alphacats.InfoSetWithAvailableActions)
		for _, action := range is.AvailableActions {
			result = append(result, encodeAction(action.Decode())...)
		}
	}

	return result
}
