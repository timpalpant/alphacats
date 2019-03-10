package model

import (
	"archive/zip"
	"fmt"
	"os"
	"path/filepath"

	"github.com/sbinet/npyio"
	"github.com/timpalpant/go-cfr/deepcfr"
)

func saveTrainingData(samples []deepcfr.Sample, directory string, batchSize int) error {
	// Write each batch as npz within the given directory.
	for batchNum := 0; batchNum*batchSize < len(samples); batchNum++ {
		batchStart := batchNum * batchSize
		batchEnd := min(batchStart+batchSize, len(samples))
		batch := samples[batchStart:batchEnd]
		batchName := fmt.Sprintf("batch_%08d.npz", batchNum)
		batchFilename := filepath.Join(directory, batchName)
		if err := saveBatch(batch, batchFilename); err != nil {
			return err
		}
	}

	return nil
}

func saveBatch(batch []deepcfr.Sample, filename string) error {
	xHistory := encodeHistories(batch)
	xHand := encodeHands(batch)
	y := encodeTargets(batch)
	sampleWeights := encodeSampleWeights(batch)
	return SaveNPZFile(filename, map[string]interface{}{
		"X_history":     xHistory,
		"X_hand":        xHand,
		"y":             y,
		"sample_weight": sampleWeights,
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
