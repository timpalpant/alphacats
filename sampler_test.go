package alphacats

import (
	"math/rand"
	"testing"
)

type randomStrategy struct {
}

func (s randomStrategy) Select(n int) int {
	return rand.Intn(n)
}

func TestSampleHistory(t *testing.T) {
	root := NewGameTree()
	s := randomStrategy{}
	SampleHistory(root, s)
}

func BenchmarkSampleHistory(b *testing.B) {
	root := NewGameTree()
	s := randomStrategy{}
	for i := 0; i < b.N; i++ {
		SampleHistory(root, s)
	}
}
