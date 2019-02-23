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
		glog.Infof("Running CFR iteration %d", i)
		rand.Shuffle(drawPile.Len(), func(i, j int) {
			tmp := drawPile.NthCard(i)
			drawPile.SetNthCard(i, drawPile.NthCard(j))
			drawPile.SetNthCard(j, tmp)
		})
		game := alphacats.NewGame(drawPile, p0Deal, p1Deal)
		expectedValue += opt.Run(game)

		currentEV := expectedValue / float32(i)
		winRate := 0.5 + currentEV/2.0
		rps := float64(i) / time.Since(start).Seconds()
		glog.Infof("CFR iteration complete. EV: %.4g => win rate: %.3f (%.1f iter/sec)",
			currentEV, winRate, rps)
	}

	expectedValue /= float32(*iter)
	glog.Infof("Expected value is: %v", expectedValue)
}
