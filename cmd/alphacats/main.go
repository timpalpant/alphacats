package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"time"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	iter := flag.Int("iter", 100000, "Number of iterations to perform")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	opt := cfr.New(cfr.Params{
		SampleChanceNodes:     true,
		SampleOpponentActions: true,
		LinearWeighting:       true,
	})

	drawPile := cards.NewStackFromCards([]cards.Card{
		cards.Shuffle, cards.SeeTheFuture, cards.ExplodingCat,
		cards.Skip,
	})

	p0Deal := cards.NewSetFromCards([]cards.Card{
		cards.Defuse, cards.Skip, cards.Slap2x,
		cards.DrawFromTheBottom,
	})

	p1Deal := cards.NewSetFromCards([]cards.Card{
		cards.Defuse, cards.Slap1x, cards.Cat,
		cards.Skip,
	})

	var expectedValue float32
	start := time.Now()
	for i := 1; i <= *iter; i++ {
		rand.Shuffle(drawPile.Len(), func(i, j int) {
			tmp := drawPile.NthCard(i)
			drawPile.SetNthCard(i, drawPile.NthCard(j))
			drawPile.SetNthCard(j, tmp)
		})
		game := alphacats.NewGame(drawPile, p0Deal, p1Deal)
		expectedValue += opt.Run(game)

		if i%100 == 0 {
			currentEV := expectedValue / float32(i)
			winRate := 0.5 + currentEV/2.0
			rps := float64(i) / time.Since(start).Seconds()
			glog.Infof("CFR iteration %d complete. EV: %.4g => win rate: %.3f (%.1f iter/sec)",
				i, currentEV, winRate, rps)
		}
	}

	expectedValue /= float32(*iter)
	glog.Infof("Expected value is: %v", expectedValue)

	rand.Shuffle(drawPile.Len(), func(i, j int) {
		tmp := drawPile.NthCard(i)
		drawPile.SetNthCard(i, drawPile.NthCard(j))
		drawPile.SetNthCard(j, tmp)
	})
	var game cfr.GameTreeNode = alphacats.NewGame(drawPile, p0Deal, p1Deal)
	store := opt.GetPolicyStore()
	glog.Infof("Playing example game")
	playGame(game, store)
}

func playGame(game cfr.GameTreeNode, store cfr.PolicyStore) {
	for game.Type() != cfr.TerminalNode {
		game.BuildChildren()
		if game.Type() == cfr.ChanceNode {
			glog.Infof("Chance node, randomly sampling child")
			game = game.SampleChild()
		} else {
			glog.Info(game)
			policy := store.GetPolicy(game)
			strat := policy.GetAverageStrategy()
			selected, p := sampleOne(strat)
			game = game.GetChild(selected)
			lastAction := game.(*alphacats.GameNode).LastAction()
			glog.Infof("=> Selected action %v with probability %.2f",
				lastAction, p)
		}
	}

	glog.Infof("Player %v wins!", game.Player())
}

func sampleOne(p []float32) (int, float32) {
	x := float32(rand.Float64())
	var cumProb float32
	for i := 0; i < len(p); i++ {
		cumProb += p[i]
		if cumProb > x {
			return i, p[i]
		}
	}

	n := len(p) - 1
	return n, p[n]
}
