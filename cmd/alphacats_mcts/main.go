// This version of alphacats uses Smooth UCT MCTS only.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/mcts"
	"github.com/timpalpant/go-cfr/sampling"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

var stdin = bufio.NewReader(os.Stdin)

type RunParams struct {
	DeckType          string
	NumMCTSIterations int
	SamplingParams    SamplingParams
	Temperature       float64
}

type SamplingParams struct {
	Seed  int64
	C     float64
	Gamma float64
	Eta   float64
	D     float64
}

func main() {
	var params RunParams
	flag.IntVar(&params.NumMCTSIterations, "iter", 100000, "Number of MCTS iterations to perform")
	flag.Float64Var(&params.Temperature, "temperature", 0.1,
		"Temperature used when selecting actions during play")
	flag.Int64Var(&params.SamplingParams.Seed, "sampling.seed", 1234, "Random seed")
	flag.Float64Var(&params.SamplingParams.C, "sampling.c", 1.75,
		"Exploration factor C used in MCTS search")
	flag.Float64Var(&params.SamplingParams.Gamma, "sampling.gamma", 0.1,
		"Mixing factor Gamma used in Smooth UCT search")
	flag.Float64Var(&params.SamplingParams.Eta, "sampling.eta", 0.9,
		"Mixing factor eta used in Smooth UCT search")
	flag.Float64Var(&params.SamplingParams.D, "sampling.d", 0.001,
		"Mixing factor d used in Smooth UCT search")

	flag.Parse()

	rand.Seed(params.SamplingParams.Seed)
	go http.ListenAndServe("localhost:4123", nil)

	deck := cards.CoreDeck.AsSlice()
	cardsPerPlayer := 4
	optimizer := mcts.NewSmoothUCT(
		float32(params.SamplingParams.C), float32(params.SamplingParams.Gamma),
		float32(params.SamplingParams.Eta), float32(params.SamplingParams.D),
		float32(params.Temperature))
	for i := 0; ; i++ {
		deal := alphacats.NewRandomDeal(deck, cardsPerPlayer)
		playGame(optimizer, params, deal)
	}
}

func simulate(optimizer *mcts.SmoothUCT, beliefs *alphacats.BeliefState, n int) {
	var wg sync.WaitGroup
	nWorkers := runtime.NumCPU()
	nPerWorker := n / nWorkers
	glog.Infof("Simulating %d games in %d workers", nWorkers*nPerWorker, nWorkers)
	for worker := 0; worker < nWorkers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < nPerWorker; k++ {
				game := beliefs.SampleDeterminization()
				optimizer.Run(game)
			}
		}()
	}

	wg.Wait()
}

func playGame(policy *mcts.SmoothUCT, params RunParams, deal alphacats.Deal) {
	var game cfr.GameTreeNode = alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)

	glog.Infof("Building initial info set")
	infoSet := game.(*alphacats.GameNode).GetInfoSet(gamestate.Player1)
	beliefs := alphacats.NewBeliefState(policy.GetPolicy, infoSet)
	glog.Infof("Initial info set has %d game states", beliefs.Len())
	simulate(policy, beliefs, params.NumMCTSIterations)

	for game.Type() != cfr.TerminalNodeType {
		if game.Type() == cfr.ChanceNodeType {
			var p float64
			game, p = game.SampleChild()
			glog.Infof("[chance] Sampled child node with probability %v", p)
		} else if game.Player() == 0 {
			is := game.InfoSet(game.Player()).(*alphacats.AbstractedInfoSet)
			glog.Infof("[player] Your turn. %d cards remaining in draw pile.",
				game.(*alphacats.GameNode).GetDrawPile().Len())
			glog.Infof("[player] Hand: %v, Choices:", is.Hand)
			for i, action := range is.AvailableActions {
				glog.Infof("%d: %v", i, action)
			}

			selected := prompt("Which action? ")
			game = game.GetChild(selected)
			lastAction := game.(*alphacats.GameNode).LastAction()
			glog.Infof("[player] Chose to %v", lastAction)
		} else {
			simulate(policy, beliefs, params.NumMCTSIterations)
			p := policy.GetPolicy(game)
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
			lastAction := game.(*alphacats.GameNode).LastAction()
			glog.Infof("[strategy] Chose to %v with probability %v: %v",
				hidePrivateInfo(lastAction), p[selected], p)
			glog.V(4).Infof("[strategy] Action result was: %v", lastAction)
		}

		glog.Info("Propagating beliefs")
		beliefs.Update(game.(*alphacats.GameNode).GetInfoSet(gamestate.Player1))
	}

	glog.Info("GAME OVER")
	if game.Player() == 0 {
		glog.Info("You win!")
	} else {
		glog.Info("Computer wins!")
	}

	glog.Info("Game history:")
	h := game.(*alphacats.GameNode).GetHistory()
	for i, action := range h.AsSlice() {
		glog.Infof("%d: %v", i, action)
	}
}

func prompt(msg string) int {
	for {
		fmt.Print(msg)
		result, err := stdin.ReadString('\n')
		if err != nil {
			panic(err)
		}

		result = strings.TrimRight(result, "\n")
		i, err := strconv.Atoi(result)
		if err != nil {
			glog.Errorf("Invalid selection: %v", result)
			continue
		}

		return i
	}
}

func hidePrivateInfo(a gamestate.Action) gamestate.Action {
	a.PositionInDrawPile = 0
	a.CardsSeen = [3]cards.Card{}
	return a
}
