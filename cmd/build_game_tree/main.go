package main

import (
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	_ "net/http/pprof"

	"github.com/timpalpant/go-cfr"

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

	game := alphacats.NewGame()
	fmt.Printf("%d terminal nodes.\n", cfr.CountTerminalNodes(game.RootNode()))
}
