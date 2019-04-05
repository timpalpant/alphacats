package npyio

import (
	"bufio"
	"os"

	"github.com/klauspost/compress/zip"
)

func MakeNPZ(output string, entries map[string][]float32) error {
	f, err := os.Create(output)
	if err != nil {
		return err
	}
	defer f.Close()

	b := bufio.NewWriter(f)
	defer b.Flush()
	z := zip.NewWriter(b)
	defer z.Close()

	for name, data := range entries {
		// Custom header so that we do not compress (Method 0=Store).
		hdr := zip.FileHeader{Name: name}
		w, err := z.CreateHeader(&hdr)
		if err != nil {
			return err
		}

		if err := Write(w, data); err != nil {
			return err
		}
	}

	return nil
}
