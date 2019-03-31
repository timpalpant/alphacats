package npyio

import (
	"bufio"
	"io"
	"os"

	"github.com/klauspost/compress/zip"
)

func MakeNPZ(npyFiles map[string]io.Reader, output string) error {
	f, err := os.Create(output)
	if err != nil {
		return err
	}
	defer f.Close()

	b := bufio.NewWriter(f)
	defer b.Flush()
	z := zip.NewWriter(b)
	defer z.Close()

	for name, r := range npyFiles {
		w, err := z.Create(name)
		if err != nil {
			return err
		}

		if _, err := io.Copy(w, r); err != nil {
			return err
		}
	}

	return nil
}
