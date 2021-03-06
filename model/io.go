package model

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model/internal/npyio"
)

func saveTrainingData(batch []Sample, filename string) error {
	nSamples := len(batch)

	histories := make([]float32, 0, nSamples*gamestate.MaxNumActions*numActionFeatures)
	hands := make([]float32, 0, nSamples*(3*numCardsInDeck))
	drawPiles := make([]float32, 0, nSamples*maxCardsInDrawPile*cards.NumTypes)
	outputMasks := make([]float32, 0, nSamples*outputDimension)
	yPolicy := make([]float32, 0, nSamples*outputDimension)
	yValue := make([]float32, 0, nSamples)

	history := newOneHotHistory()
	var hand [numCardsInDeck]float32
	drawPile := newOneHotDrawPile()
	var outputMask [outputDimension]float32
	var policy [outputDimension]float32
	for _, sample := range batch {
		is := sample.InfoSet
		if len(is.AvailableActions) != len(sample.Policy) {
			panic(fmt.Errorf("InfoSet has %d actions but policy has %d: %v",
				len(is.AvailableActions), len(sample.Policy), is.AvailableActions))
		}

		EncodeHistory(is.PublicHistory, history)
		for _, row := range history {
			histories = append(histories, row...)
		}
		encodeHand(is.Hand, hand[:])
		hands = append(hands, hand[:]...)
		encodeHand(is.P0PlayedCards, hand[:])
		hands = append(hands, hand[:]...)
		encodeHand(is.P1PlayedCards, hand[:])
		hands = append(hands, hand[:]...)
		encodeDrawPile(is.DrawPile, drawPile)
		for _, row := range drawPile {
			drawPiles = append(drawPiles, row...)
		}
		encodeOutputMask(is.DrawPile.Len(), is.AvailableActions, outputMask[:])
		outputMasks = append(outputMasks, outputMask[:]...)

		encodeOutputs(is.DrawPile.Len(), is.AvailableActions, sample.Policy, policy[:])
		yPolicy = append(yPolicy, policy[:]...)
		yValue = append(yValue, sample.Value)
	}

	return npyio.MakeNPZ(filename, map[string][]float32{
		"X_history":     histories,
		"X_hands":       hands,
		"X_drawpile":    drawPiles,
		"X_output_mask": outputMasks,
		"Y_policy":      yPolicy,
		"Y_value":       yValue,
	})
}

func min(i, j int) int {
	if i < j {
		return i
	}

	return j
}
