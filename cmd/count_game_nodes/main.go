// Script to estimate the number of nodes touched in an external sampling run.
package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	seed := flag.Int64("seed", 123, "Seed for random game")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4124", nil)

	deck := cards.CoreDeck.AsSlice()
	game := alphacats.NewRandomGame(deck, 4)
	rng := rand.New(rand.NewSource(rand.Int63()))
	total := countNodes(game, rng, 0)
	glog.Infof("%d nodes in game", total)
}

func countNodes(node cfr.GameTreeNode, rng *rand.Rand, depth int) int {
	switch node.Type() {
	case cfr.ChanceNode:
		child, _ := node.SampleChild()
		total := countNodes(child, rng, depth+1) + 1
		node.Close()
		return total
	case cfr.PlayerNode:
		if node.Player() == 0 {
			total := 1
			for i := 0; i < node.NumChildren(); i++ {
				child := node.GetChild(i)
				total += countNodes(child, rng, depth+1)
			}

			node.Close()
			return total
		} else {
			selected := rng.Intn(node.NumChildren())
			child := node.GetChild(selected)
			total := countNodes(child, rng, depth+1) + 1
			node.Close()
			return total
		}
	}

	return 1
}
