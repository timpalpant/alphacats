package model

import (
	"archive/zip"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/sbinet/npyio"
	"github.com/timpalpant/go-cfr/deepcfr"
)

// Writing to npz files is slow, but we have to buffer the (large)
// one-hot encoded feature tensors in memory before they are written out.
// Each batch of 4096 samples requires ~200MB of memory, so this works
// out to ~1.6GB required for I/O.
const maxConcurrentIOWorkers = 8

func saveTrainingData(samples []deepcfr.Sample, directory string, batchSize int) error {
	// Write each batch as npz within the given directory.
	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, maxConcurrentIOWorkers)
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
	return retErr
}

func saveBatch(batch []deepcfr.Sample, filename string) error {
	return SaveNPZFile(filename, map[string]interface{}{
		"X_history":     encodeHistories(batch),
		"X_hand":        encodeHands(batch),
		"y":             encodeTargets(batch),
		"sample_weight": encodeSampleWeights(batch),
	})
}

func SaveNPZFile(filename string, data map[string]interface{}) error {
	// Open npz (zip) file.
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	z := zip.NewWriter(f)
	defer z.Close()

	// Write each npy entry into the npz file.
	for name, entry := range data {
		w, err := z.Create(name)
		if err != nil {
			return err
		}

		if err := npyio.Write(w, entry); err != nil {
			return err
		}
	}

	return nil
}

func min(i, j int) int {
	if i < j {
		return i
	}

	return j
}
