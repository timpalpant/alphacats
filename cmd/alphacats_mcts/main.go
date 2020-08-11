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
		playGame(optimizer, params, deck, cardsPerPlayer)
	}
}

func simulate(optimizer *mcts.SmoothUCT, game cfr.GameTreeNode, drawPileConstraint cards.Stack, p1Deal cards.Set, n int) {
	infoSet := game.InfoSet(game.Player()).(*alphacats.InfoSetWithAvailableActions).InfoSet

	gamesRemaining.Add(int64(n))
	var wg sync.WaitGroup
	nWorkers := runtime.NumCPU()
	nPerWorker := n / nWorkers
	glog.Infof("Simulating %d games in %d workers", nWorkers*nPerWorker, nWorkers)
	for worker := 0; worker < nWorkers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < nPerWorker; k++ {
				game := sampleGame(drawPileConstraint, p1Deal, infoSet)

				gamesInProgress.Add(1)
				optimizer.Run(game)
				gamesInProgress.Add(-1)
				gamesRemaining.Add(-1)
				numTraversals.Add(1)
			}
		}()
	}

	wg.Wait()
}

func validateDeal(deal alphacats.Deal, drawPile cards.Stack, p1Deal cards.Set) {
	if deal.P1Deal != p1Deal {
		panic(fmt.Errorf("Invalid random game: got p1 deal %v, expected %v", deal.P1Deal, p1Deal))
	}

	allCards := deal.DrawPile.ToSet()
	allCards.AddAll(deal.P0Deal)
	allCards.AddAll(deal.P1Deal)
	expected := cards.CoreDeck
	expected.Add(cards.Defuse)
	expected.Add(cards.Defuse)
	expected.Add(cards.Defuse)
	expected.Add(cards.ExplodingCat)
	if allCards != expected {
		panic(fmt.Errorf("Invalid random game: got deck %v, expected %v", allCards, expected))
	}

	for i := 0; i < drawPile.Len(); i++ {
		nthCard := drawPile.NthCard(i)
		if nthCard == cards.Unknown {
			continue
		}

		if deal.DrawPile.NthCard(i) != nthCard {
			panic(fmt.Errorf("Invalid random game: got draw pile %v, expected %v", deal.DrawPile, drawPile))
		}
	}
}

func sampleGame(drawPileConstraint cards.Stack, p1Deal cards.Set, infoSet gamestate.InfoSet) *alphacats.GameNode {
	for i := 1; ; i++ {
		deal := alphacats.NewRandomDealWithConstraints(drawPileConstraint, p1Deal)
		validateDeal(deal, drawPileConstraint, p1Deal)
		game := alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
		game = walkForwardUntil(game, infoSet)
		if game != nil {
			return game
		}

		if i%10000 == 0 {
			glog.Warningf("Failed to find matching state for %d iterations", i)
		}
	}
}

func walkForwardUntil(game *alphacats.GameNode, target gamestate.InfoSet) *alphacats.GameNode {
	infoSet := game.GetInfoSet(gamestate.Player(game.Player()))
	if infoSet.History.Len() == target.History.Len() {
		if infoSet.Hand == target.Hand {
			return game.Clone()
		} else {
			return nil
		}
	}

	defer game.Close()
	for i := 0; i < game.NumChildren(); i++ {
		child := game.GetChild(i).(*alphacats.GameNode)
		infoSet := child.GetInfoSet(target.Player)
		if infoSet.History.Len() > target.History.Len() {
			continue
		}

		// Check whether child info set matches target.
		isMatch := true
		for j := 0; j < infoSet.History.Len(); j++ {
			if infoSet.History.GetPacked(j) != target.History.GetPacked(j) {
				isMatch = false
				break
			}
		}

		// If it did, recurse until we reach a game with equal history length.
		if isMatch {
			if game := walkForwardUntil(child, target); game != nil {
				return game
			}
		}
	}

	// No match is possible rooted at this game.
	return nil
}

func playGame(policy *mcts.SmoothUCT, params RunParams, deck []cards.Card, cardsPerPlayer int) {
	deal := alphacats.NewRandomDeal(deck, cardsPerPlayer)
	var game cfr.GameTreeNode = alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
	drawPileConstraint := cards.NewStack()
	drawPilePos := 0
	hasShuffled := false
	for game.Type() != cfr.TerminalNodeType {
		var lastAction gamestate.Action
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
			lastAction = game.(*alphacats.GameNode).LastAction()
			if lastAction.Type == gamestate.DrawCard {
				if lastAction.CardsSeen[0] == cards.ExplodingCat && !hasShuffled {
					drawPileConstraint.SetNthCard(drawPilePos, cards.ExplodingCat)
				}

				drawPilePos++
			} else if lastAction.Type == gamestate.PlayCard && lastAction.Card == cards.Shuffle {
				hasShuffled = true
			}
			glog.Infof("[player] Chose to %v", lastAction)
		} else {
			simulate(policy, game, drawPileConstraint, deal.P1Deal, params.NumMCTSIterations)
			p := policy.GetPolicy(game, float32(params.Temperature))
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
			lastAction = game.(*alphacats.GameNode).LastAction()
			glog.Infof("[strategy] Chose to %v with probability %v: %v",
				hidePrivateInfo(lastAction), p[selected], p)
			glog.V(4).Infof("[strategy] Action result was: %v", lastAction)
			if lastAction.Type == gamestate.DrawCard && !hasShuffled {
				drawnCard := lastAction.CardsSeen[0]
				drawPileConstraint.SetNthCard(drawPilePos, drawnCard)
				drawPilePos++
			} else if lastAction.Type == gamestate.PlayCard {
				if lastAction.Card == cards.Shuffle {
					hasShuffled = true
				} else if lastAction.Card == cards.DrawFromTheBottom && !hasShuffled {
					drawPileConstraint.SetNthCard(12, lastAction.CardsSeen[0])
				} else if lastAction.Card == cards.SeeTheFuture && !hasShuffled {
					for i, c := range lastAction.CardsSeen {
						drawPileConstraint.SetNthCard(drawPilePos+i, c)
					}
				}
			}
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
	fmt.Print(msg)
	result, err := stdin.ReadString('\n')
	if err != nil {
		panic(err)
	}

	result = strings.TrimRight(result, "\n")
	i, err := strconv.Atoi(result)
	if err != nil {
		panic(err)
	}

	return i
}

func hidePrivateInfo(a gamestate.Action) gamestate.Action {
	a.PositionInDrawPile = 0
	a.CardsSeen = [3]cards.Card{}
	return a
}
