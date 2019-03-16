// Generate and save samples of exploding kitten position for analysis in Python.
package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"

	"github.com/golang/glog"
	gzip "github.com/klauspost/pgzip"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	strat0 := flag.String("strategy0", "", "File with policy for player 0")
	strat1 := flag.String("strategy1", "", "File with policy for player 1")
	numGames := flag.Int("num_games", 10000, "Number of random games to play")
	flag.Parse()

	go http.ListenAndServe("localhost:4123", nil)

	deck := cards.TestDeck.AsSlice()
	cardsPerPlayer := (len(deck) / 2) - 1
	policy0 := mustLoadPolicy(*strat0)
	policy1 := mustLoadPolicy(*strat1)

	glog.Infof("Playing %d games", *numGames)
	var p0Wins int
	for i := 0; i < *numGames; i++ {
		game := alphacats.NewRandomGame(deck, cardsPerPlayer)
		winner := playGame(policy0, policy1, game)
		if winner == 0 {
			p0Wins++
		}
	}

	winRate := float64(p0Wins) / float64(*numGames)
	glog.Infof("Player 0 won %d (%.3f %%) of games", p0Wins, 100*winRate)
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

	policy, err := cfr.LoadStrategyTable(r)
	if err != nil {
		glog.Fatal(err)
	}

	return policy
}

func playGame(policy0, policy1 cfr.StrategyProfile, game cfr.GameTreeNode) int {
	for game.Type() != cfr.TerminalNode {
		if game.Type() == cfr.ChanceNode {
			game, _ = cfr.SampleChanceNode(game)
		} else if game.Player() == 0 {
			strategy := policy0.GetStrategy(game).GetAverageStrategy()
			selected := sampleStrategy(strategy)
			game = game.GetChild(selected)
		} else {
			strategy := policy1.GetStrategy(game).GetAverageStrategy()
			selected := sampleStrategy(strategy)
			game = game.GetChild(selected)
		}
	}

	return int(game.Player())
}

func sampleStrategy(strat []float32) int {
	x := rand.Float32()
	var cumProb float32
	for i, p := range strat {
		cumProb += p
		if cumProb > x {
			return i
		}
	}

	// Shouldn't ever happen unless probability distribution does not sum to 1.
	return len(strat) - 1
}
