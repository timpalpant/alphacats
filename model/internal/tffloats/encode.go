// Package tffloats constructs *tf.Tensors from []float32 slices,
// avoiding reflection.
package tffloats

import (
	"encoding/binary"
	"math"
	"unsafe"
)

func New1DTensor(v []float32) []byte {
	buf := make([]byte, 4*len(v))
	encodeF32s(v, buf)
	return buf
}

func New2DTensor(v [][]float32) []byte {
	d0, d1 := len(v), len(v[0])
	nElem := d0 * d1
	buf := make([]byte, 4*nElem)
	for i, vi := range v {
		encodeF32s(vi, buf[4*i*d1:])
	}

	return buf
}

func New3DTensor(v [][][]float32) []byte {
	d0, d1, d2 := len(v), len(v[0]), len(v[0][0])
	nElem := d0 * d1 * d2
	buf := make([]byte, 4*nElem)
	for i, vi := range v {
		for j, vj := range vi {
			startIdx := 4 * ((i*d1 + j) * d2)
			encodeF32s(vj, buf[startIdx:])
		}
	}

	return buf
}

func encodeF32s(v []float32, buf []byte) {
	for i, x := range v {
		bits := math.Float32bits(x)
		nativeEndian.PutUint32(buf[4*i:], bits)
	}
}

// nativeEndian is the byte order for the local platform. Used to send back and
// forth Tensors with the C API. We test for endianness at runtime because
// some architectures can be booted into different endian modes.
var nativeEndian binary.ByteOrder

func init() {
	buf := [2]byte{}
	*(*uint16)(unsafe.Pointer(&buf[0])) = uint16(0xABCD)

	switch buf {
	case [2]byte{0xCD, 0xAB}:
		nativeEndian = binary.LittleEndian
	case [2]byte{0xAB, 0xCD}:
		nativeEndian = binary.BigEndian
	default:
		panic("Could not determine native endianness.")
	}
}
