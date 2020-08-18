// Generate and save samples of exploding kitten position for analysis in Python.
package main

import (
	"flag"
	"io/ioutil"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"sync"
	"sync/atomic"

	"github.com/golang/glog"
	gzip "github.com/klauspost/pgzip"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"
	"github.com/timpalpant/go-cfr/sampling"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/model"
)

func main() {
	strat0 := flag.String("strategy0", "", "File with policy for player 0")
	strat1 := flag.String("strategy1", "", "File with policy for player 1")
	numGames := flag.Int("num_games", 10000, "Number of random games to play")
	seed := flag.Int64("seed", 1234, "Random seed")
	parallel := flag.Bool("parallel", false, "Play games in parallel (DeepCFR only)")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	deck := cards.TestDeck.AsSlice()
	cardsPerPlayer := (len(deck) / 2) - 1
	var wg sync.WaitGroup
	var policy0, policy1 cfr.StrategyProfile
	wg.Add(2)
	go func() {
		policy0 = mustLoadTabularPolicy(*strat0)
		wg.Done()
	}()
	go func() {
		policy1 = mustLoadTabularPolicy(*strat1)
		wg.Done()
	}()
	wg.Wait()

	glog.Infof("Playing %d games", *numGames)
	var p0Wins int64
	for i := 0; i < *numGames; i++ {
		deal := alphacats.NewRandomDeal(deck, cardsPerPlayer)
		game := alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)

		playGame := func(i int) {
			// Alternate which player goes first.
			if i%2 == 0 {
				if winner := playGame(policy0, policy1, game); winner == 0 {
					atomic.AddInt64(&p0Wins, 1)
				}
			} else {
				if winner := playGame(policy1, policy0, game); winner == 1 {
					atomic.AddInt64(&p0Wins, 1)
				}
			}

			wg.Done()
		}

		wg.Add(1)
		if *parallel {
			go playGame(i)
		} else {
			playGame(i)
		}

		if i%(*numGames/10) == 0 {
			glog.Infof("Played %d games", i)
		}
	}

	wg.Wait()
	winRate := float64(p0Wins) / float64(*numGames)
	glog.Infof("Policy 0 won %d (%.3f %%) of games", p0Wins, 100*winRate)
	glog.Infof("Policy 1 won %d (%.3f %%) of games", int64(*numGames)-p0Wins, 100*(1-winRate))
}

func mustLoadTabularPolicy(filename string) cfr.StrategyProfile {
	policy := cfr.NewPolicyTable(cfr.DiscountParams{})
	mustLoadPolicy(filename, policy)
	return policy
}

func mustLoadDeepCFRPolicy(filename string) cfr.StrategyProfile {
	lstm := model.NewLSTM(model.Params{})
	buffers := []deepcfr.Buffer{}
	baselineBuffers := []deepcfr.Buffer{}
	policy := deepcfr.NewVRSingleDeepCFR(lstm, buffers, baselineBuffers)
	mustLoadPolicy(filename, policy)
	return policy
}

func mustLoadPolicy(filename string, policy cfr.StrategyProfile) {
	glog.Infof("Loading strategy from: %v", filename)
	f, err := os.Open(filename)
	if err != nil {
		glog.Fatal(err)
	}
	defer f.Close()

	r, err := gzip.NewReader(f)
	if err != nil {
		glog.Fatal(err)
	}

	buf, err := ioutil.ReadAll(r)
	if err != nil {
		glog.Fatal(err)
	}

	if err := policy.UnmarshalBinary(buf); err != nil {
		glog.Fatal(err)
	}
}

func playGame(policy0, policy1 cfr.StrategyProfile, game cfr.GameTreeNode) int {
	for game.Type() != cfr.TerminalNodeType {
		if game.Type() == cfr.ChanceNodeType {
			game, _ = game.SampleChild()
		} else if game.Player() == 0 {
			p := policy0.GetPolicy(game).GetAverageStrategy()
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
		} else {
			p := policy1.GetPolicy(game).GetAverageStrategy()
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
		}
	}

	return int(game.Player())
}
