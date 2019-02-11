package main

import (
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	_ "net/http/pprof"

	"github.com/timpalpant/alphacats"
)

type testStrategy struct {
}

func (s testStrategy) Select(n int) int {
	return 0 // Always choose first option
}

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	flag.Parse()
	go http.ListenAndServe("localhost:4123", nil)

	rand.Seed(*seed)

	tree := alphacats.NewGameTree()
	fmt.Printf("%d terminal nodes.\n", alphacats.CountTerminalNodes(tree))

	s := testStrategy{}
	history := alphacats.SampleHistory(tree, s)
	for i, turn := range history {
		fmt.Printf("Turn %d: %s\n", i, turn)
	}
}
