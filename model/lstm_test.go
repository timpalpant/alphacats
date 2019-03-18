package model

import (
	"fmt"
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

const testModel = "testdata/savedmodel"

// with request batching:
// BenchmarkPredict-24            	500	    3210621 ns/op
// BenchmarkPredictParallel-24    	3000     409844 ns/op
//
// without request batching:
// BenchmarkPredict-24            	500	    2741763 ns/op
// BenchmarkPredictParallel-24    	500	    2848236 ns/op
//
// with async request batching:
// BenchmarkPredict-24            	500	    2686052 ns/op
// BenchmarkPredictParallel-24    	500	    3134419 ns/op
func BenchmarkPredict(b *testing.B) {
	deck := cards.CoreDeck.AsSlice()
	game := alphacats.NewRandomGame(deck, 4)
	is := game.InfoSet(0)
	model, err := LoadTrainedLSTM(testModel)
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	model.Predict(is, 6) // One time setup cost.

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Predict(is, 6)
	}
}

func BenchmarkPredictParallel(b *testing.B) {
	deck := cards.CoreDeck.AsSlice()
	game := alphacats.NewRandomGame(deck, 4)
	is := game.InfoSet(0)
	model, err := LoadTrainedLSTM(testModel)
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	model.Predict(is, 6) // One time setup cost.

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			model.Predict(is, 6)
		}
	})
}

// BenchmarkPredict/batchSize=1-24         	     500	   2943487 ns/op
// enchmarkPredict/batchSize=8-24         	     500	   3044852 ns/op
// BenchmarkPredict/batchSize=16-24        	     500	   3111497 ns/op
// BenchmarkPredict/batchSize=32-24        	     500	   3596349 ns/op
// BenchmarkPredict/batchSize=64-24        	     500	   3918713 ns/op
// BenchmarkPredict/batchSize=128-24       	     300	   4974051 ns/op
// BenchmarkPredict/batchSize=256-24       	     200	   8301496 ns/op
func BenchmarkBatchSize(b *testing.B) {
	deck := cards.CoreDeck.AsSlice()
	game := alphacats.NewRandomGame(deck, 4)
	is := game.InfoSet(0).(*gamestate.InfoSet)
	history := EncodeHistory(is.History)
	hand := encodeHand(is.Hand)

	opts := &tf.SessionOptions{Config: tfConfig}
	model, err := tf.LoadSavedModel(testModel, []string{graphTag}, opts)
	if err != nil {
		b.Fatal(err)
	}
	defer model.Session.Close()

	for _, batchSize := range []int{1, 8, 16, 32, 64, 128, 256} {
		runBatch(b, model, history, hand, batchSize)
	}
}

func runBatch(b *testing.B, model *tf.SavedModel, history [][]float32, hand []float32, batchSize int) {
	historyBatch := make([][][]float32, batchSize)
	handBatch := make([][]float32, batchSize)
	for i := range historyBatch {
		historyBatch[i] = history
		handBatch[i] = hand
	}

	historyTensor, err := tf.NewTensor(historyBatch)
	if err != nil {
		b.Fatal(err)
	}

	handTensor, err := tf.NewTensor(handBatch)
	if err != nil {
		b.Fatal(err)
	}

	// There is some expensive one-time cost so make one prediction so the
	// remainder of the numbers are comparable.
	_, err = model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("history").Output(0): historyTensor,
			model.Graph.Operation("hand").Output(0):    handTensor,
		},
		[]tf.Output{
			model.Graph.Operation("output/BiasAdd").Output(0),
		},
		nil,
	)

	if err != nil {
		b.Fatal(err)
	}

	name := fmt.Sprintf("batchSize=%d", batchSize)
	b.Run(name, func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := model.Session.Run(
				map[tf.Output]*tf.Tensor{
					model.Graph.Operation("history").Output(0): historyTensor,
					model.Graph.Operation("hand").Output(0):    handTensor,
				},
				[]tf.Output{
					model.Graph.Operation("output/BiasAdd").Output(0),
				},
				nil,
			)

			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
