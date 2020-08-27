// Generate training samples for PSRO network bootstrap by playing
// games with Smooth UCT search.
package main

import (
	"bufio"
	"encoding/gob"
	"expvar"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/mcts"
	"github.com/timpalpant/go-cfr/sampling"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/model"
)

var (
	p0NumSamples      = expvar.NewInt("num_samples/player0")
	p1NumSamples      = expvar.NewInt("num_samples/player1")
	gamesPlayed       = expvar.NewInt("games_played")
	p0Wins            = expvar.NewInt("num_wins/player0")
	p1Wins            = expvar.NewInt("num_wins/player1")
	searchesPerformed = expvar.NewInt("searches_performed")
)

type RunParams struct {
	Deck           []cards.Card
	CardsPerPlayer int

	OutputDir           string
	NumTrainingSamples  int
	CheckpointInterval  time.Duration
	MaxParallelGames    int
	NumMCTSIterations   int
	MaxParallelSearches int

	SamplingParams SamplingParams
	Temperature    float64
}

type SamplingParams struct {
	Seed  int64
	C     float64
	Gamma float64
	Eta   float64
	D     float64
}

func main() {
	params := RunParams{
		Deck:           cards.CoreDeck.AsSlice(),
		CardsPerPlayer: 4,
	}
	flag.StringVar(&params.OutputDir, "output_dir", "models/bootstrap-training-data",
		"Output directory to save generated training data to")
	flag.IntVar(&params.NumTrainingSamples, "num_training_samples", 200000,
		"Maximum number of training samples to keep")
	flag.DurationVar(&params.CheckpointInterval, "checkpoint_interval", 30*time.Minute,
		"How often to write out collected samples to output diretory")
	flag.IntVar(&params.MaxParallelGames, "max_parallel_games", 4,
		"Number of games to run in parallel")
	flag.IntVar(&params.NumMCTSIterations, "search_iter", 100000,
		"Number of MCTS iterations to perform per move")
	flag.IntVar(&params.MaxParallelSearches, "max_parallel_searches", runtime.NumCPU(),
		"Number of searches per game to run in parallel")
	flag.Float64Var(&params.Temperature, "temperature", 1.0,
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

	glog.Info("Performing bootstrap with SmoothUCT search")
	var mx sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, params.MaxParallelGames)
	trainingData := [][]model.Sample{
		make([]model.Sample, 0, params.NumTrainingSamples/2),
		make([]model.Sample, 0, params.NumTrainingSamples/2),
	}
	lastCheckpoint := time.Now()
	var iter, nTotal int
	for ; nTotal < params.NumTrainingSamples; iter++ {
		if time.Since(lastCheckpoint) > params.CheckpointInterval {
			mx.Lock()
			if err := saveCheckpoint(params.OutputDir, iter, trainingData); err != nil {
				glog.Fatal(err)
			}
			trainingData[0] = trainingData[0][:0]
			trainingData[1] = trainingData[1][:0]
			mx.Unlock()
		}

		sem <- struct{}{}
		wg.Add(1)
		go func() {
			defer func() {
				wg.Done()
				<-sem
			}()

			deal := alphacats.NewRandomDeal(params.Deck, params.CardsPerPlayer)
			game := alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
			glog.Infof("Playing game with ~%d search iterations", params.NumMCTSIterations)
			samples := playGame(game, params)
			glog.Infof("Collected %d samples", len(samples))
			mx.Lock()
			defer mx.Unlock()
			for i, s := range samples {
				glog.V(1).Infof("Sample %d: %v", i, s)
				trainingData[s.InfoSet.Player] = append(trainingData[s.InfoSet.Player], s)
				nTotal++
			}

			gamesPlayed.Add(1)
			p0NumSamples.Set(int64(len(trainingData[0])))
			p1NumSamples.Set(int64(len(trainingData[1])))
		}()
	}

	wg.Wait()

	// Save final checkpoint.
	if err := saveCheckpoint(params.OutputDir, iter, trainingData); err != nil {
		glog.Fatal(err)
	}
}

func playGame(game cfr.GameTreeNode, params RunParams) []model.Sample {
	search := mcts.NewSmoothUCT(
		float32(params.SamplingParams.C), float32(params.SamplingParams.Gamma),
		float32(params.SamplingParams.Eta), float32(params.SamplingParams.D),
		float32(params.Temperature))
	p0InfoSet := game.(*alphacats.GameNode).GetInfoSet(gamestate.Player0)
	p0Beliefs := alphacats.NewBeliefState(search.GetPolicy, p0InfoSet)
	p1InfoSet := game.(*alphacats.GameNode).GetInfoSet(gamestate.Player1)
	p1Beliefs := alphacats.NewBeliefState(search.GetPolicy, p1InfoSet)
	beliefs := []*alphacats.BeliefState{p0Beliefs, p1Beliefs}

	var samples []model.Sample
	for game.Type() != cfr.TerminalNodeType {
		if game.Type() == cfr.ChanceNodeType {
			game, _ = game.SampleChild()
		} else {
			is := game.InfoSet(game.Player()).(*alphacats.AbstractedInfoSet)
			// We perform simulations of both players so that the belief update distributions
			// are fairly well approximated. If we only simulate over the acting player's
			// current belief distribution, then the belief update for the other player's
			// non-real worlds may not have much data.
			simulate(search, beliefs[0], params.NumMCTSIterations, params.MaxParallelSearches)
			simulate(search, beliefs[1], params.NumMCTSIterations, params.MaxParallelSearches)
			p := search.GetPolicy(game)
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
			samples = append(samples, model.Sample{
				InfoSet: *is,
				Policy:  p,
			})
		}

		beliefs[0].Update(game.(*alphacats.GameNode).GetInfoSet(gamestate.Player0))
		beliefs[1].Update(game.(*alphacats.GameNode).GetInfoSet(gamestate.Player1))
	}

	if game.Player() == 0 {
		p0Wins.Add(1)
	} else {
		p1Wins.Add(1)
	}

	for i, s := range samples {
		if s.InfoSet.Player == gamestate.Player(game.Player()) {
			samples[i].Value = 1.0
		} else {
			samples[i].Value = -1.0
		}
	}

	return samples
}

func saveCheckpoint(outputDir string, iter int, trainingData [][]model.Sample) error {
	for player, playerSamples := range trainingData {
		if err := saveTrainingSamples(outputDir, iter, player, playerSamples); err != nil {
			return err
		}
	}

	return nil
}

func saveTrainingSamples(outputDir string, iter int, player int, samples []model.Sample) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return err
	}

	filename := filepath.Join(outputDir, fmt.Sprintf("player_%d.%4d.samples", player, iter))
	glog.Infof("Saving player %d training samples to: %v", player, filename)
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	enc := gob.NewEncoder(w)
	if err := enc.Encode(samples); err != nil {
		f.Close()
		return err
	}

	w.Flush()
	return f.Close()
}

func simulate(search *mcts.SmoothUCT, beliefs *alphacats.BeliefState, n, nParallel int) {
	var wg sync.WaitGroup
	nWorkers := min(n, nParallel)
	nPerWorker := n / nWorkers
	for worker := 0; worker < nWorkers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(rand.Int63()))
			for k := 0; k < nPerWorker; k++ {
				game := beliefs.SampleDeterminization()
				search.Run(rng, game)
				searchesPerformed.Add(1)
			}
		}()
	}

	wg.Wait()
}

func min(i, j int) int {
	if i < j {
		return i
	}
	return j
}
