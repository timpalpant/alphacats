// This version of alphacats uses lazy one-sided IS-MCTS, in a PSRO framework.
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
	"github.com/timpalpant/alphacats/model"
)

var stdin = bufio.NewReader(os.Stdin)

var (
	searchesByLevel  = expvar.NewMap("searches_by_level")
	cacheHitsByLevel = expvar.NewMap("cache_hits_by_level")
)

type RunParams struct {
	OracleDepth         int
	NumMCTSIterations   int
	MaxParallelSearches int

	SamplingParams SamplingParams
	Temperature    float64
}

type SamplingParams struct {
	Seed int64
	C    float64
}

func main() {
	var params RunParams
	flag.IntVar(&params.OracleDepth, "oracle_depth", 5,
		"Number of FP oracles to train")
	flag.IntVar(&params.NumMCTSIterations, "search_iter", 50000,
		"Number of MCTS iterations to perform per move")
	flag.IntVar(&params.MaxParallelSearches, "max_parallel_searches", runtime.NumCPU(),
		"Number of searches per game to run in parallel")
	flag.Float64Var(&params.Temperature, "temperature", 1.0,
		"Temperature used when selecting actions during play")
	flag.Int64Var(&params.SamplingParams.Seed, "sampling.seed", 123,
		"Random seed")
	flag.Float64Var(&params.SamplingParams.C, "sampling.c", 1.75,
		"Exploration factor C used in MCTS search")

	flag.Parse()

	rand.Seed(params.SamplingParams.Seed)
	go http.ListenAndServe("localhost:4124", nil)

	deck := cards.CoreDeck.AsSlice()
	cardsPerPlayer := 4
	for i := 0; ; i++ {
		deal := alphacats.NewRandomDeal(deck, cardsPerPlayer)
		playGame(params, deal)
	}
}

type RecursiveSearchPolicy struct {
	player  gamestate.Player
	beliefs *alphacats.BeliefState
	search  *mcts.OneSidedISMCTS

	level       string
	temperature float32
	numSearches int
	numWorkers  int

	// Indicates infoset has already been searched.
	mx    sync.Mutex
	cache map[string]struct{}
}

func (r *RecursiveSearchPolicy) GetPolicy(node cfr.GameTreeNode) []float32 {
	is := node.InfoSet(int(r.player)).Key()
	r.mx.Lock()
	defer r.mx.Unlock()
	if _, ok := r.cache[is]; ok {
		cacheHitsByLevel.Add(r.level, 1)
		return r.search.GetPolicy(node, r.temperature)
	}

	searchesByLevel.Add(r.level, 1)
	beliefs := r.beliefs.Clone()
	glog.V(1).Info("Propagating beliefs")
	beliefs.Update(node.(*alphacats.GameNode).GetInfoSet(r.player))

	var wg sync.WaitGroup
	nPerWorker := r.numSearches / r.numWorkers
	glog.V(1).Infof("Simulating %d games in %d workers", r.numWorkers*nPerWorker, r.numWorkers)
	for worker := 0; worker < r.numWorkers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < nPerWorker; k++ {
				game := beliefs.SampleDeterminization()
				r.search.Run(game)
			}
		}()
	}

	wg.Wait()
	r.cache[is] = struct{}{}
	return r.search.GetPolicy(node, r.temperature)
}

func playGame(params RunParams, deal alphacats.Deal) {
	var game cfr.GameTreeNode = alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)

	glog.Infof("Building initial info set")
	policies := []mcts.Policy{&model.UniformRandomPolicy{}, &model.UniformRandomPolicy{}}
	for i := 0; i < 2*params.OracleDepth; i++ {
		player := i % 2
		infoSet := game.(*alphacats.GameNode).GetInfoSet(gamestate.Player(player))
		opponentPolicy := policies[1-player]
		level := fmt.Sprintf("p%d_o%d", player, i/2)
		policies[player] = &RecursiveSearchPolicy{
			player:      gamestate.Player(player),
			beliefs:     alphacats.NewBeliefState(opponentPolicy.GetPolicy, infoSet),
			search:      mcts.NewOneSidedISMCTS(player, opponentPolicy, mcts.NewRandomRollout(1), float32(params.SamplingParams.C)),
			level:       level,
			temperature: float32(params.Temperature),
			numSearches: params.NumMCTSIterations,
			numWorkers:  params.MaxParallelSearches,
			cache:       make(map[string]struct{}),
		}
		searchesByLevel.Set(level, expvar.NewInt(level+"_searches"))
		cacheHitsByLevel.Set(level, expvar.NewInt(level+"_cache_hits"))
	}
	policy := policies[1].(*RecursiveSearchPolicy)

	for game.Type() != cfr.TerminalNodeType {
		nodeType := game.Type()
		if nodeType == cfr.ChanceNodeType {
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
			p := policy.GetPolicy(game)
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
