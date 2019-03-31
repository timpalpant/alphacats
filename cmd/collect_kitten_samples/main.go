// Generate and save samples of exploding kitten position for analysis in Python.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model"
)

const maxCardsInDrawPile = 13

type Sample struct {
	History                 gamestate.History
	ExplodingKittenPosition int
}

func main() {
	numBatches := flag.Int("batches", 1000, "Number of batches of samples to collect")
	batchSize := flag.Int("batch_size", 4096, "Number of samples in each batch file")
	output := flag.String("output", "", "Output directory to save collected batches of samples to")
	flag.Parse()

	go http.ListenAndServe("localhost:4123", nil)

	if err := os.MkdirAll(*output, 0777); err != nil {
		glog.Fatal(err)
	}

	deck := cards.CoreDeck.AsSlice()
	for i := 0; i < *numBatches; i++ {
		glog.Infof("Collecting %d samples", *batchSize)
		samples := make([]Sample, 0, *batchSize)
		for len(samples) < *batchSize {
			game := alphacats.NewRandomGame(deck, 4)
			samples = append(samples, collectSamples(game)...)
		}

		batchName := fmt.Sprintf("batch_%08d", i)
		batchFilename := filepath.Join(*output, batchName+".npz")
		if err := saveSamples(samples, batchFilename); err != nil {
			glog.Fatal(err)
		}
	}
}

func collectSamples(game cfr.GameTreeNode) []Sample {
	// Play out the given game to the end, choosing actions uniformly randomly.
	var terminalHistory []Sample
	for game.Type() != cfr.TerminalNode {
		if game.Type() == cfr.ChanceNode {
			game, _ = game.SampleChild()
		} else {
			// All samples are collected from the POV of player 0.
			is := game.InfoSet(0).(*gamestate.InfoSet)
			drawPile := game.(*alphacats.GameNode).GetDrawPile()
			sample := Sample{
				History:                 is.History,
				ExplodingKittenPosition: getKittenPosition(drawPile),
			}

			// Kitten may not be in the deck if this is a MustDefuse node
			// i.e. currently in the player's hand waiting to be replaced.
			if sample.ExplodingKittenPosition != -1 {
				terminalHistory = append(terminalHistory, sample)
			}

			// Randomly choose a player action.
			selected := rand.Intn(game.NumChildren())
			game = game.GetChild(selected)
		}
	}

	return terminalHistory
}

func getKittenPosition(drawPile cards.Stack) int {
	result := -1

	i := 0
	drawPile.Iter(func(card cards.Card) {
		if card == cards.ExplodingCat {
			result = i
		}

		i++
	})

	return result
}

func saveSamples(samples []Sample, output string) error {
	glog.Infof("Encoding samples and targets")
	X := encodeSamples(samples)
	y := encodeTargets(samples)

	glog.Infof("Saving %d samples to %v", len(samples), output)
	return model.SaveNPZFile(output, map[string]interface{}{
		"X": X,
		"y": y,
	})
}

func encodeSamples(samples []Sample) []float32 {
	var result []float32
	for _, sample := range samples {
		X := model.EncodeHistory(sample.History)
		result = append(result, ravel(X)...)
	}

	return result
}

func encodeTargets(samples []Sample) []float32 {
	var result []float32
	for _, sample := range samples {
		y := oneHot(maxCardsInDrawPile, sample.ExplodingKittenPosition)
		result = append(result, y...)
	}

	return result
}

func oneHot(n, j int) []float32 {
	result := make([]float32, n)
	result[j] = 1.0
	return result
}

func ravel(X [][]float32) []float32 {
	if len(X) == 0 {
		return nil
	}

	r := len(X)
	c := len(X[0])
	result := make([]float32, 0, r*c)
	for _, row := range X {
		if len(row) != c {
			panic(fmt.Errorf("row has invalid length: got %d, expected %d",
				len(row), c))
		}

		result = append(result, row...)
	}

	return result
}
