package model

import (
	"fmt"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model/internal/npyio"
)

func saveTrainingData(samples []deepcfr.Sample, directory string, batchSize int, maxNumWorkers int) error {
	// Normalize sample weights to have mean 1 for each sampled infoset.
	// Only the relative weights within an infoset matter for correctness in expectation.
	normalizeSampleWeights(samples)

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

func saveBatch(batch []deepcfr.Sample, filename string) error {
	nSamples := 0
	for _, sample := range batch {
		nSamples += len(sample.Advantages)
	}

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

		if len(is.AvailableActions) != len(sample.Advantages) {
			panic(fmt.Errorf("Sample has %d actions but %d advantages: sample=%v, is=%v",
				len(is.AvailableActions), len(sample.Advantages), sample, is))
		}

		EncodeHistory(is.History, history)
		encodeHand(is.Hand, hand[:])
		w := float32(int((sample.Weight + 1.0) / 2.0))
		for _, action := range is.AvailableActions {
			for _, row := range history {
				histories = append(histories, row...)
			}

			hands = append(hands, hand[:]...)
			encodeAction(action, oneHotAction[:])
			actions = append(actions, oneHotAction[:]...)
			sampleWeights = append(sampleWeights, w)
		}

		y = append(y, sample.Advantages...)
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

type sampleWeightStats struct {
	total float64
	count int
}

func (s sampleWeightStats) mean() float32 {
	return float32(s.total / float64(s.count))
}

func normalizeSampleWeights(samples []deepcfr.Sample) {
	glog.V(1).Infof("Bucketing samples by infoset")
	byInfoSet := make(map[string]sampleWeightStats)
	for _, s := range samples {
		stats := byInfoSet[string(s.InfoSet)]
		stats.total += float64(s.Weight)
		stats.count++
		byInfoSet[string(s.InfoSet)] = stats
	}

	glog.V(1).Infof("Normalizing sample weights by infoset")
	counts := make(map[int]int)
	for i, s := range samples {
		stats := byInfoSet[string(s.InfoSet)]
		counts[stats.count]++
		samples[i].Weight /= stats.mean() // NB: modify the slice
	}

	glog.V(1).Info("Infoset duplication:")
	for k := range sortedKeys(counts) {
		glog.V(1).Infof("%d: %d", k, counts[k])
	}
}

func sortedKeys(counts map[int]int) []int {
	result := make([]int, 0, len(counts))
	for k := range counts {
		result = append(result, k)
	}

	sort.Ints(result)
	return result
}
