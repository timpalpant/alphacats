package model

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model/internal/npyio"
)

func saveTrainingData(samples []*deepcfr.ExperienceTuple, directory string, batchSize int, maxNumWorkers int) error {
	// Normalize sample weights to have mean 1 for each sampled infoset.
	// Only the relative weights within an infoset matter for correctness in expectation.
	normalizeSampleWeights(samples)
	rand.Shuffle(len(samples), func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})

	// Write each batch as npz within the given directory.
	glog.V(1).Infof("Writing batches to %v", directory)
	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, maxNumWorkers)
	start := time.Now()
	var retErr error
	for batchNum := 0; batchNum*batchSize < len(samples); batchNum++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(batchNum int) {
			defer func() { <-sem }()
			defer wg.Done()

			batchStart := batchNum * batchSize
			batchEnd := min(batchStart+batchSize, len(samples))
			batch := samples[batchStart:batchEnd]
			batchName := fmt.Sprintf("batch_%08d.npz", batchNum)
			batchFilename := filepath.Join(directory, batchName)
			glog.V(2).Infof("Saving batch %d (%d samples) to %v",
				batchNum, len(batch), batchFilename)
			if err := saveBatch(batch, batchFilename); err != nil {
				mu.Lock()
				defer mu.Unlock()
				if retErr == nil {
					retErr = err
				}
			}
		}(batchNum)
	}

	wg.Wait()

	elapsed := time.Since(start)
	sps := float64(len(samples)) / elapsed.Seconds()
	glog.V(1).Infof("Finished saving training data (took: %v, %.1f samples/sec)", elapsed, sps)
	return retErr
}

func saveBatch(batch []*deepcfr.ExperienceTuple, filename string) error {
	nSamples := len(batch)

	histories := make([]float32, 0, nSamples*gamestate.MaxNumActions*numActionFeatures)
	hands := make([]float32, 0, nSamples*cards.NumTypes)
	actions := make([]float32, 0, nSamples*numActionFeatures)
	y := make([]float32, 0, nSamples)
	sampleWeights := make([]float32, 0, nSamples)

	var is alphacats.InfoSetWithAvailableActions
	history := newOneHotHistory()
	var hand [cards.NumTypes]float32
	var oneHotAction [numActionFeatures]float32
	for _, sample := range batch {
		if err := is.UnmarshalBinary(sample.InfoSet); err != nil {
			return err
		}

		EncodeHistory(is.History, history)
		for _, row := range history {
			histories = append(histories, row...)
		}
		encodeHand(is.Hand, hand[:])
		hands = append(hands, hand[:]...)
		encodeAction(is.AvailableActions[sample.Action], oneHotAction[:])
		actions = append(actions, oneHotAction[:]...)
		sampleWeights = append(sampleWeights, sample.Weight)

		y = append(y, sample.Value)
	}

	return npyio.MakeNPZ(filename, map[string][]float32{
		"X_history":     histories,
		"X_hand":        hands,
		"X_action":      actions,
		"y":             y,
		"sample_weight": sampleWeights,
	})
}

func min(i, j int) int {
	if i < j {
		return i
	}

	return j
}

func normalizeSampleWeights(samples []*deepcfr.ExperienceTuple) {
	var mean float32
	for _, s := range samples {
		mean += s.Weight / float32(len(samples))
	}

	for i, s := range samples {
		s.Weight /= mean
		samples[i] = s
	}
}
