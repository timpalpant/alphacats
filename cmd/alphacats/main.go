package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"time"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/tree"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	vanillaCFR := cfr.NewVanilla()
	alphacats.EnumerateGames(func(drawPile cards.Stack, p0Deal, p1Deal cards.Set) {
		game := alphacats.NewGame(drawPile, p0Deal, p1Deal)
		total := 0
		start := time.Now()
		tree.Visit(game, func(node cfr.GameTreeNode) {
			total++
			if total%10000000 == 0 {
				nps := float64(total) / time.Since(start).Seconds()
				glog.Infof("Visited %d nodes (%.1f nodes/sec)", total, nps)
			}
		})

		glog.Infof("%d terminal nodes", total)
		expectedValue := vanillaCFR.Run(game)
		glog.Infof("Expected value is: %v", expectedValue)
	})
}
