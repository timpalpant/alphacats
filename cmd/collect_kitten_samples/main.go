// Generate and save samples of exploding kitten position for analysis in Python.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"

	"github.com/golang/glog"
	gzip "github.com/klauspost/pgzip"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model"
)

type Sample struct {
	InfoSet                 gamestate.InfoSet
	NumCardsInDrawPile      int
	ExplodingKittenPosition int
}

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	numBatches := flag.Int("batches", 100, "Number of batches of samples to collect")
	batchSize := flag.Int("batch_size", 1000000, "Number of samples in each batch file")
	output := flag.String("output", "", "Output directory to save collected batches of samples to")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	if err := os.MkdirAll(*output, 0777); err != nil {
		glog.Fatal(err)
	}

	for i := 0; i < *numBatches; i++ {
		glog.Infof("Collecting %d samples", *batchSize)
		samples := make([]Sample, 0)
		for len(samples) < *batchSize {
			game := alphacats.NewRandomGame()
			samples = append(samples, collectSamples(game)...)
		}
		samples = samples[:*batchSize]

		batchName := fmt.Sprintf("batch_%08d.pb.gz", i)
		filename := filepath.Join(*output, batchName)
		glog.Infof("Saving batch of samples to %v", filename)
		if err := saveSamples(samples, filename); err != nil {
			glog.Fatal(err)
		}
	}
}

func collectSamples(game *alphacats.GameNode) []Sample {
	var result []Sample
	for game.Type() != cfr.TerminalNode {
		if game.Type() == cfr.ChanceNode {
			game = game.SampleChild().(*alphacats.GameNode)
		} else {
			is := game.InfoSet(game.Player()).(gamestate.InfoSet)
			drawPile := game.GetDrawPile()
			sample := Sample{
				InfoSet:                 is,
				NumCardsInDrawPile:      drawPile.Len(),
				ExplodingKittenPosition: getKittenPosition(drawPile),
			}

			// Kitten may not be in the deck if this is a MustDefuse node
			// i.e. currently in the player's hand waiting to be replaced.
			if sample.ExplodingKittenPosition != -1 {
				result = append(result, sample)
			}

			// Randomly choose a player action.
			selected := rand.Intn(game.NumChildren())
			game = game.GetChild(selected).(*alphacats.GameNode)
		}
	}

	return result
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
	f, err := os.Create(output)
	if err != nil {
		return err
	}
	defer f.Close()

	w := gzip.NewWriter(f)
	defer w.Close()

	var varintBuf [binary.MaxVarintLen64]byte
	for _, sample := range samples {
		// First serialize proto with player's InfoSet.
		pb := infosetToProto(sample.InfoSet)
		buf, err := pb.Marshal()
		if err != nil {
			return err
		}

		n := binary.PutUvarint(varintBuf[:], uint64(len(buf)))
		if _, err := w.Write(varintBuf[:n]); err != nil {
			return err
		}

		if _, err := w.Write(buf); err != nil {
			return err
		}

		// Then the number of cards currently remaining in the draw pile.
		n = binary.PutUvarint(varintBuf[:], uint64(sample.NumCardsInDrawPile))
		if _, err := w.Write(varintBuf[:n]); err != nil {
			return err
		}

		// Then the current position of the exploding kitten in the draw pile.
		n = binary.PutUvarint(varintBuf[:], uint64(sample.ExplodingKittenPosition))
		if _, err := w.Write(varintBuf[:n]); err != nil {
			return err
		}
	}

	return nil
}

func infosetToProto(is gamestate.InfoSet) *model.InfoSet {
	return &model.InfoSet{
		History: historyToProto(is.History),
		Hand:    cardsToProto(is.Hand),
	}
}

func historyToProto(h gamestate.History) []*model.Action {
	result := make([]*model.Action, h.Len())
	for i, action := range h.AsSlice() {
		result[i] = actionToProto(action)
	}

	return result
}

func actionToProto(action gamestate.Action) *model.Action {
	var cardsSeen []model.Card
	for _, card := range action.CardsSeen {
		if card != cards.Unknown {
			cardsSeen = append(cardsSeen, model.Card(card))
		}
	}

	return &model.Action{
		Player:             int32(action.Player),
		Type:               model.Action_Type(action.Type),
		Card:               model.Card(action.Card),
		PositionInDrawPile: int32(action.PositionInDrawPile),
		CardsSeen:          cardsSeen,
	}
}

func cardsToProto(hand cards.Set) []model.Card {
	result := make([]model.Card, hand.Len())
	for i, card := range hand.AsSlice() {
		result[i] = model.Card(card)
	}

	return result
}
