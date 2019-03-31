// Script to estimate the number of nodes touched in an external sampling run.
package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"time"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	seed := flag.Int64("seed", 123, "Seed for random game")
	flag.Parse()

	go http.ListenAndServe("localhost:4124", nil)

	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		for _ = range ticker.C {
			logMemUsage()
		}
	}()

	deck := cards.CoreDeck.AsSlice()
	game := alphacats.NewRandomGame(deck, 4)
	rng := rand.New(rand.NewSource(*seed))
	sampledActions := make(map[string]int)
	total := countNodes(game, rng, sampledActions)
	glog.Infof("%d sampled actions, %d nodes in game", len(sampledActions), total)
	logMemUsage()
}

func countNodes(node cfr.GameTreeNode, rng *rand.Rand, sampledActions map[string]int) int {
	switch node.Type() {
	case cfr.ChanceNode:
		child, _ := node.SampleChild()
		total := countNodes(child, rng, sampledActions) + 1
		node.Close()
		return total
	case cfr.PlayerNode:
		if node.Player() == 0 {
			total := 1
			for i := 0; i < node.NumChildren(); i++ {
				child := node.GetChild(i)
				total += countNodes(child, rng, sampledActions)
			}

			node.Close()
			return total
		} else {
			selected := rng.Intn(node.NumChildren())
			child := node.GetChild(selected)
			total := countNodes(child, rng, sampledActions) + 1
			node.Close()
			return total
		}
	}

	return 1
}

// PrintMemUsage outputs the current, total and OS memory being used. As well as the number
// of garage collection cycles completed.
func logMemUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	// For info on each, see: https://golang.org/pkg/runtime/#MemStats
	glog.Infof("Alloc = %v MiB", bToMb(m.Alloc))
	glog.Infof("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
	glog.Infof("\tSys = %v MiB", bToMb(m.Sys))
	glog.Infof("\tNumGC = %v\n", m.NumGC)
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}
