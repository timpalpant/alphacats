package model

import (
	"bufio"
	"fmt"
	"io"
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

// Writing to npz files is slow, but we have to buffer the (large)
// one-hot encoded feature tensors in memory before they are written out.
// Each batch of 4096 samples requires ~100MB of memory, so this works
// out to ~1.6GB required for I/O.
const maxConcurrentIOWorkers = 16

func saveTrainingData(samples []deepcfr.Sample, directory string, batchSize int) error {
	// Write each batch as npz within the given directory.
	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, maxConcurrentIOWorkers)
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

	npyFilenames := map[string]int{
		"X_history":     nSamples * gamestate.MaxNumActions * numActionFeatures,
		"X_hand":        nSamples * cards.NumTypes,
		"X_action":      nSamples * numActionFeatures,
		"y":             nSamples,
		"sample_weight": nSamples,
	}

	npyFiles := make(map[string]*npyio.File, len(npyFilenames))
	for filename, numElements := range npyFilenames {
		f, err := npyio.Create(filepath.Join(tmpDir, filename), numElements)
		if err != nil {
			return err
		}
		defer f.Close()

		npyFiles[filename] = f
	}

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
				if err := npyFiles["X_history"].Append(row...); err != nil {
					return err
				}
			}

			if err := npyFiles["X_hand"].Append(hand...); err != nil {
				return err
			}

			if err := npyFiles["sample_weight"].Append(w); err != nil {
				return err
			}

			if err := npyFiles["X_action"].Append(encodeAction(action)...); err != nil {
				return err
			}
		}

		npyFiles["y"].Append(sample.Advantages...)
	}

	toZip := make(map[string]io.Reader, len(npyFiles))
	for name, npyF := range npyFiles {
		if err := npyF.Close(); err != nil {
			return err
		}

		f, err := os.Open(filepath.Join(tmpDir, name))
		if err != nil {
			return err
		}
		defer f.Close()

		toZip[name] = bufio.NewReader(f)
	}

	return npyio.MakeNPZ(toZip, filename)
}

func min(i, j int) int {
	if i < j {
		return i
	}

	return j
}
