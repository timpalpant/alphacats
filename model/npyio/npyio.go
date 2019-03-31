package npyio

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

var order = binary.LittleEndian

type File struct {
	f   *os.File
	buf *bufio.Writer
}

func Create(filename string, numElements int) (*File, error) {
	f, err := os.Create(filename)
	if err != nil {
		return nil, err
	}

	buf := bufio.NewWriter(f)
	err = writeHeader(buf, numElements)
	if err != nil {
		f.Close()
		return nil, err
	}

	return &File{
		f:   f,
		buf: buf,
	}, nil
}

func (f *File) Close() error {
	f.buf.Flush()
	return f.f.Close()
}

func (f *File) Append(v ...float32) error {
	var buf [4]byte
	for _, x := range v {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(x))
		_, err := f.buf.Write(buf[:])
		if err != nil {
			return err
		}
	}
	return nil
}

// The following is adapted from: github.com/sbinet/npyio
var magic = [6]byte{'\x93', 'N', 'U', 'M', 'P', 'Y'}

const (
	majorVersion = byte(2)
	minorVersion = byte(0)
)

func writeHeader(w io.Writer, numElements int) error {
	if err := binary.Write(w, order, magic[:]); err != nil {
		return err
	}
	if err := binary.Write(w, order, majorVersion); err != nil {
		return err
	}
	if err := binary.Write(w, order, minorVersion); err != nil {
		return err
	}

	buf := new(bytes.Buffer)
	fmt.Fprintf(buf,
		"{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }",
		numElements)

	var hdrSize = 6 + len(magic)
	padding := (hdrSize + buf.Len() + 1) % 16
	if _, err := buf.Write(bytes.Repeat([]byte{'\x20'}, padding)); err != nil {
		return err
	}
	if _, err := buf.Write([]byte{'\n'}); err != nil {
		return err
	}

	buflen := int64(buf.Len())
	if err := binary.Write(w, order, uint32(buflen)); err != nil {
		return err
	}

	if n, err := io.Copy(w, buf); err != nil {
		return err
	} else if n < buflen {
		return io.ErrShortWrite
	}

	return nil
}
