package model

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model/npyio"
)

func saveTrainingData(samples []deepcfr.Sample, directory string, batchSize int, maxNumWorkers int) error {
	// Write each batch as npz within the given directory.
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
	outputDir := filepath.Dir(filename)
	tmpDir, err := ioutil.TempDir(outputDir, "npy-entries-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

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
	for _, sample := range batch {
		if err := is.UnmarshalBinary(sample.InfoSet); err != nil {
			return err
		}

		if len(is.AvailableActions) != len(sample.Advantages) {
			panic(fmt.Errorf("Sample has %d actions but %d advantages: sample=%v, is=%v",
				len(is.AvailableActions), len(sample.Advantages), sample, is))
		}

		history := EncodeHistory(is.History)
		hand := encodeHand(is.Hand)
		w := float32(int((sample.Weight + 1.0) / 2.0))
		for _, action := range is.AvailableActions {
			for _, row := range history {
				histories = append(histories, row...)
			}

			hands = append(hands, hand...)
			actions = append(actions, encodeAction(action)...)
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
