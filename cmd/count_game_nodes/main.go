// Script to estimate the number of nodes touched in an external sampling run.
package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"sync"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	seed := flag.Int64("seed", 123, "Seed for random game")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4125", nil)

	deck := cards.CoreDeck.AsSlice()
	game := alphacats.NewRandomGame(deck, 4)

	workQueue := make(chan cfr.GameTreeNode, runtime.NumCPU())
	for i := 0; i < cap(workQueue); i++ {
		rng := rand.New(rand.NewSource(rand.Int63()))
		go countNodes(workQueue)
	}

	total := countNodesParallel(game, workQueue)
	glog.Infof("%d nodes in game", total)
}

type countJob struct {
	root   cfr.GameTreeNode
	result chan int
}

func countNodesParallel(node cfr.GameTreeNode, workQueue chan cfr.GameTreeNode) int {
	switch node.Type() {
	case cfr.ChanceNodeType:
		child, _ := node.SampleChild()
		total := countNodesParallel(child, sem) + 1
		node.Close()
		return total
	case cfr.PlayerNodeType:
		total := 1
		var wg sync.WaitGroup
		var mu sync.Mutex
		for i := 0; i < node.NumChildren(); i++ {
			child := node.GetChild(i)

			sem <- struct{}{}
			wg.Add(1)
			go func() {
				rng := rand.New(rand.NewSource(rand.Int63()))
				n := countNodes(child, rng)
				mu.Lock()
				total += n
				mu.Unlock()
				<-sem
				wg.Done()
			}()
		}

		wg.Done()
		node.Close()
		return total
	}

	return 1

}

func countNodes(node cfr.GameTreeNode, rng *rand.Rand) int {
	switch node.Type() {
	case cfr.ChanceNodeType:
		child, _ := node.SampleChild()
		total := countNodes(child, rng) + 1
		node.Close()
		return total
	case cfr.PlayerNodeType:
		if node.Player() == 0 {
			total := 1
			for i := 0; i < node.NumChildren(); i++ {
				child := node.GetChild(i)
				total += countNodes(child, rng)
			}

			node.Close()
			return total
		} else {
			selected := rng.Intn(node.NumChildren())
			child := node.GetChild(selected)
			total := countNodes(child, rng) + 1
			node.Close()
			return total
		}
	}

	return 1
}
