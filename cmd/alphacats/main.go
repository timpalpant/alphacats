package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/tree"

	"github.com/timpalpant/alphacats"
)

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	iter := flag.Int("iter", 100000, "Number of iterations to perform")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	opt := cfr.New(cfr.Params{
		SampleChanceNodes: true,
	})
	var game cfr.GameTreeNode
	var expectedValue float32
	for i := 0; i < *iter; i++ {
		game = alphacats.NewRandomGame()
		expectedValue += opt.Run(game)
	}

	expectedValue /= float32(*iter)
	glog.Infof("Expected value is: %v", expectedValue)

	tree.VisitInfoSets(game, func(player int, infoSet cfr.InfoSet) {
		strat := opt.GetStrategy(player, infoSet)
		if strat != nil {
			glog.Infof("[player %d] %v: %v", player, infoSet, strat)
		}
	})
}
