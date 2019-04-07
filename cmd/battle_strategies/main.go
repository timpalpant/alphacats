// Generate and save samples of exploding kitten position for analysis in Python.
package main

import (
	"encoding/gob"
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"sync"

	"github.com/golang/glog"
	gzip "github.com/klauspost/pgzip"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/sampling"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	_ "github.com/timpalpant/alphacats/model"
)

func main() {
	strat0 := flag.String("strategy0", "", "File with policy for player 0")
	strat1 := flag.String("strategy1", "", "File with policy for player 1")
	numGames := flag.Int("num_games", 10000, "Number of random games to play")
	seed := flag.Int64("seed", 1234, "Random seed")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	deck := cards.TestDeck.AsSlice()
	cardsPerPlayer := (len(deck) / 2) - 1
	var wg sync.WaitGroup
	var policy0, policy1 cfr.StrategyProfile
	wg.Add(2)
	go func() {
		policy0 = mustLoadPolicy(*strat0)
		wg.Done()
	}()
	go func() {
		policy1 = mustLoadPolicy(*strat1)
		wg.Done()
	}()
	wg.Wait()

	glog.Infof("Playing %d games", *numGames)
	var p0Wins int
	for i := 0; i < *numGames; i++ {
		game := alphacats.NewRandomGame(deck, cardsPerPlayer)
		// Alternate which player goes first.
		if i%2 == 0 {
			winner := playGame(policy0, policy1, game)
			if winner == 0 {
				p0Wins++
			}
		} else {
			winner := playGame(policy1, policy0, game)
			if winner == 1 {
				p0Wins++
			}
		}
	}

	winRate := float64(p0Wins) / float64(*numGames)
	glog.Infof("Policy 0 won %d (%.3f %%) of games", p0Wins, 100*winRate)
	glog.Infof("Policy 1 won %d (%.3f %%) of games", 1-p0Wins, 100*(1-winRate))
}

func mustLoadPolicy(filename string) cfr.StrategyProfile {
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

	var policy cfr.StrategyProfile
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&policy); err != nil {
		glog.Fatal(err)
	}

	return policy
}

func playGame(policy0, policy1 cfr.StrategyProfile, game cfr.GameTreeNode) int {
	for game.Type() != cfr.TerminalNode {
		if game.Type() == cfr.ChanceNode {
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
