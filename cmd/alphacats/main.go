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
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	iter := flag.Int("iter", 100000, "Number of iterations to perform")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	drawPile := cards.NewStackFromCards([]cards.Card{
		cards.ExplodingCat, cards.Slap1x, cards.Slap2x,
	})
	p0Hand := cards.NewSetFromCards([]cards.Card{
		cards.Cat, cards.Defuse,
	})
	p1Hand := cards.NewSetFromCards([]cards.Card{
		cards.Skip, cards.Defuse,
	})
	opt := cfr.New(cfr.Params{})
	var game cfr.GameTreeNode
	var expectedValue float64
	for i := 0; i < *iter; i++ {
		rand.Shuffle(drawPile.Len(), func(i, j int) {
			tmp := drawPile.NthCard(i)
			drawPile.SetNthCard(i, drawPile.NthCard(j))
			drawPile.SetNthCard(j, tmp)
		})

		game = alphacats.NewGame(drawPile, p0Hand, p1Hand)
		expectedValue += opt.Run(game)
	}

	expectedValue /= float64(*iter)
	glog.Infof("Expected value is: %v", expectedValue)

	tree.VisitInfoSets(game, func(player int, infoSet cfr.InfoSet) {
		strat := opt.GetStrategy(player, infoSet)
		if strat != nil {
			glog.Infof("[player %d] %v: %v", player, infoSet, strat)
		}
	})
}
