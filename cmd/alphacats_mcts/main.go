package main

import (
	"bufio"
	"expvar"
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

var (
	gamesInProgress = expvar.NewInt("games_in_progress")
	gamesRemaining  = expvar.NewInt("games_remaining")
	numTraversals   = expvar.NewInt("num_traversals")
)

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

func getDeck(deckType string) (deck []cards.Card, cardsPerPlayer int) {
	switch deckType {
	case "test":
		deck = cards.TestDeck.AsSlice()
		cardsPerPlayer = (len(deck) / 2) - 1
	case "core":
		deck = cards.CoreDeck.AsSlice()
		cardsPerPlayer = 4
	default:
		panic(fmt.Errorf("unknown deck type: %v", deckType))
	}

	return deck, cardsPerPlayer
}

func main() {
	var params RunParams
	flag.StringVar(&params.DeckType, "decktype", "test", "Type of deck to use (core, test)")
	flag.IntVar(&params.NumMCTSIterations, "iter", 100, "Number of MCTS iterations to perform")
	flag.Float64Var(&params.Temperature, "temperature", 0.5,
		"Temperature used when selecting actions during play")
	flag.Int64Var(&params.SamplingParams.Seed, "sampling.seed", 123, "Random seed")
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

	deck, cardsPerPlayer := getDeck(params.DeckType)
	optimizer := mcts.NewSmoothUCT(float32(params.SamplingParams.C),
		float32(params.SamplingParams.Gamma), float32(params.SamplingParams.Eta),
		float32(params.SamplingParams.D))
	for i := 0; ; i++ {
		deal := alphacats.NewRandomDeal(deck, cardsPerPlayer)
		playGame(optimizer, params, deal)
	}
}

func simulate(optimizer *mcts.SmoothUCT, beliefs *beliefState, n int) {
	var wg sync.WaitGroup
	nWorkers := runtime.NumCPU()
	nPerWorker := n / nWorkers
	glog.Infof("Simulating %d games in %d workers", nWorkers*nPerWorker, nWorkers)
	for worker := 0; worker < nWorkers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(rand.Int63()))
			for k := 0; k < nPerWorker; k++ {
				selected := rng.Intn(len(beliefs.states))
				game := beliefs.states[selected].Clone()
				optimizer.Run(game)
			}
		}()
	}

	wg.Wait()
}

func simulateRandomGames(optimizer *mcts.SmoothUCT, n int) {
	var wg sync.WaitGroup
	nWorkers := runtime.NumCPU()
	nPerWorker := n / nWorkers
	glog.Infof("Simulating %d games in %d workers", nWorkers*nPerWorker, nWorkers)
	for worker := 0; worker < nWorkers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			deck := cards.CoreDeck.AsSlice()
			for k := 0; k < nPerWorker; k++ {
				deal := alphacats.NewRandomDeal(deck, 4)
				game := alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
				optimizer.Run(game)
			}
		}()
	}

	wg.Wait()

}

func playGame(policy *mcts.SmoothUCT, params RunParams, deal alphacats.Deal) {
	var game cfr.GameTreeNode = alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
	simulateRandomGames(policy, runtime.NumCPU()*params.NumMCTSIterations)

	glog.Infof("Building initial info set")
	beliefs := makeInitialBeliefState(deal)
	glog.Infof("Initial info set has %d game states", len(beliefs.states))

	for game.Type() != cfr.TerminalNodeType {
		if game.Type() == cfr.ChanceNodeType {
			var p float64
			game, p = game.SampleChild()
			glog.Infof("[chance] Sampled child node with probability %v", p)
		} else if game.Player() == 0 {
			is := game.InfoSet(game.Player()).(*alphacats.InfoSetWithAvailableActions)
			glog.Infof("[player] Your turn. %d cards remaining in draw pile.",
				game.(*alphacats.GameNode).GetDrawPile().Len())
			glog.Infof("[player] Hand: %v, Choices:", is.InfoSet.Hand)
			for i, action := range is.AvailableActions {
				glog.Infof("%d: %v", i, action)
			}

			selected := prompt("Which action? ")
			game = game.GetChild(selected)
			lastAction := game.(*alphacats.GameNode).LastAction()
			glog.Infof("[player] Chose to %v", lastAction)

			glog.Info("Propogating beliefs")
			beliefs = propogateBeliefs(policy, beliefs, hidePrivateInfo(is.AvailableActions[selected]), float32(params.Temperature))
			glog.Infof("Infoset now has %d states", len(beliefs.states))
		} else {
			simulate(policy, beliefs, params.NumMCTSIterations)
			p := policy.GetPolicy(game, float32(params.Temperature))
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
			lastAction := game.(*alphacats.GameNode).LastAction()
			glog.Infof("[strategy] Chose to %v with probability %v: %v",
				hidePrivateInfo(lastAction), p[selected], p)
			glog.V(4).Infof("[strategy] Action result was: %v", lastAction)
		}
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

type partialState struct {
	nDrawPileCards     int
	fixedDrawPileCards cards.Stack

	nPlayer0Cards     int
	knownPlayer0Cards cards.Set

	nPlayer1Cards     int
	knownPlayer1Cards cards.Set

	discardPile cards.Set
}

type beliefState struct {
	states     []*alphacats.GameNode
	reachProbs []float32
}

func propogateBeliefs(policy *mcts.SmoothUCT, bs *beliefState, action gamestate.Action, temperature float32) *beliefState {
	var states []*alphacats.GameNode
	var reachProbs []float32
	for i, game := range bs.states {
		p := policy.GetPolicy(game, temperature)
		is := game.InfoSet(game.Player()).(*alphacats.InfoSetWithAvailableActions)
		for j, a := range is.AvailableActions {
			if hidePrivateInfo(a) == action {
				states = append(states, game.GetChild(j).(*alphacats.GameNode))
				reachProbs = append(reachProbs, p[j]*bs.reachProbs[i])
			}
		}
	}

	return &beliefState{states, reachProbs}
}

// Return all game states consistent with player 1 initial deal.
func makeInitialBeliefState(deal alphacats.Deal) *beliefState {
	p1Hand := deal.P1Deal
	p1Hand.Remove(cards.Defuse)

	remaining := cards.CoreDeck
	remaining.RemoveAll(p1Hand)
	emptyDrawPile := cards.NewStack()
	var states []*alphacats.GameNode
	seen := make(map[cards.Set]struct{})
	enumerateDealsHelper(remaining, cards.NewSet(), p1Hand.Len(), func(p0Hand cards.Set) {
		if _, ok := seen[p0Hand]; ok {
			return
		}

		seen[p0Hand] = struct{}{}
		p0Deal := p0Hand
		p0Deal.Add(cards.Defuse)
		p1Deal := p1Hand
		p1Deal.Add(cards.Defuse)
		game := alphacats.NewGame(emptyDrawPile, p0Deal, p1Deal)
		states = append(states, game)
	})

	return &beliefState{
		states:     states,
		reachProbs: uniformDistribution(len(states)),
	}
}

func uniformDistribution(n int) []float32 {
	result := make([]float32, n)
	for i := range result {
		result[i] = 1.0 / float32(n)
	}
	return result
}

func enumerateDealsHelper(deck cards.Set, result cards.Set, n int, cb func(deal cards.Set)) {
	if n == 0 {
		cb(result)
		return
	}

	deck.Iter(func(card cards.Card, count uint8) {
		remaining := deck
		remaining.Remove(card)
		newResult := result
		newResult.Add(card)
		enumerateDealsHelper(remaining, newResult, n-1, cb)
	})
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
