// Package tffloats constructs *tf.Tensors from []float32 slices,
// avoiding reflection.
package tffloats

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

func EncodeF32s(v []float32, buf []byte) {
	if len(buf) < 4*len(v) {
		panic(fmt.Errorf("trying to encode %d float32s into buffer of size %d",
			len(v), len(buf)))
	}

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
