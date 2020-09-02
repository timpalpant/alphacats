// This version of alphacats uses one-sided IS-MCTS with a NN
// to guide search, in a PSRO framework.
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
	gamesRemaining    = expvar.NewInt("games_remaining")
	searchesPerformed = expvar.NewInt("searches_performed")
)

type RunParams struct {
	Deck           []cards.Card
	CardsPerPlayer int

	BootstrapSamplesDir string
	NumGamesPerEpoch    int
	MaxParallelGames    int
	NumMCTSIterations   int
	MaxParallelSearches int

	SamplingParams   SamplingParams
	Temperature      float64
	SampleBufferSize int
	RetrainInterval  int

	ModelParams         model.Params
	PredictionCacheSize int
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
	flag.StringVar(&params.BootstrapSamplesDir, "bootstrap_samples_dir", "models/bootstrap-training-data",
		"Directory with bootstrap training data for initial model")
	flag.IntVar(&params.NumGamesPerEpoch, "games_per_epoch", 5000,
		"Number of games to play each epoch")
	flag.IntVar(&params.MaxParallelGames, "max_parallel_games", runtime.NumCPU(),
		"Number of games to run in parallel")
	flag.IntVar(&params.NumMCTSIterations, "search_iter", 1000,
		"Number of MCTS iterations to perform per move")
	flag.IntVar(&params.MaxParallelSearches, "max_parallel_searches", runtime.NumCPU(),
		"Number of searches per game to run in parallel")
	flag.IntVar(&params.SampleBufferSize, "sample_buffer_size", 500000,
		"Maximum number of training samples to keep")
	flag.IntVar(&params.RetrainInterval, "retrain_interval", 10000,
		"How many samples to collect before retraining.")
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
	flag.StringVar(&params.ModelParams.OutputDir, "model.output_dir", "models",
		"Output directory for trained models")
	flag.IntVar(&params.ModelParams.NumEncodingWorkers, "model.encoding_workers", 4,
		"Maximum number of workers for training data encoding")
	flag.IntVar(&params.ModelParams.NumPredictionWorkers, "model.num_predict_workers", 4,
		"Number of background prediction workers")
	flag.IntVar(&params.ModelParams.MaxInferenceBatchSize, "model.predict_batch_size", 2048,
		"Maximum batch size for prediction")
	flag.IntVar(&params.PredictionCacheSize, "prediction_cache_size", 100000,
		"Size of LRU prediction cache per model")

	flag.Parse()

	rand.Seed(params.SamplingParams.Seed)
	go http.ListenAndServe("localhost:4123", nil)

	policies := loadPolicy(params)
	if policies[0].Len() == 0 {
		bootstrap(policies[0], 0, params)
	}
	if policies[1].Len() == 0 {
		bootstrap(policies[1], 1, params)
	}

	for epoch := 0; ; epoch++ {
		player := epoch % 2
		glog.Infof("Starting epoch %d: Playing %d games to train approximate best response for player %d",
			epoch, params.NumGamesPerEpoch, player)
		runEpoch(policies, player, params)
	}
}

func bootstrap(policy *model.MCTSPSRO, player int, params RunParams) {
	trainingData, err := filepath.Glob(filepath.Join(params.BootstrapSamplesDir, fmt.Sprintf("player_%d.*.samples", player)))
	if err != nil || len(trainingData) > 0 {
		glog.Infof("Training initial model with bootstrap data from %d files", len(trainingData))
		for _, file := range trainingData {
			samples, err := loadTrainingSamples(file)
			if err != nil {
				glog.Fatalf("Error loading training samples from %v: %v", file, err)
			}

			for _, s := range samples {
				policy.AddSample(s)
			}
		}

		policy.TrainNetwork()
		policy.AddCurrentExploiterToModel()
	} else {
		glog.Warningf("Didn't find any bootstrap data for player %d. Starting with uniform random model", player)
		policy.AddModel(&model.UniformRandomPolicy{})
	}
}

func loadTrainingSamples(filename string) ([]model.Sample, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	dec := gob.NewDecoder(r)
	var samples []model.Sample
	err = dec.Decode(&samples)
	return samples, err
}

func runEpoch(policies [2]*model.MCTSPSRO, player int, params RunParams) {
	gamesRemaining.Add(int64(params.NumGamesPerEpoch))
	var wg sync.WaitGroup
	sem := make(chan struct{}, params.MaxParallelGames)
	policy := policies[player]
	opponent := policies[1-player]
	ismcts := mcts.NewOneSidedISMCTS(player, policy, float32(params.SamplingParams.C), float32(params.Temperature))
	for i := 0; i < params.NumGamesPerEpoch; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer func() {
				gamesRemaining.Add(-1)
				wg.Done()
				<-sem
			}()
			deal := alphacats.NewRandomDeal(params.Deck, params.CardsPerPlayer)
			game := alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
			opponentPolicy := opponent.SamplePolicy()
			glog.Infof("Playing game with ~%d search iterations", params.NumMCTSIterations)
			samples := playGame(game, ismcts, opponentPolicy, player, params)
			glog.Infof("Collected %d samples", len(samples))
			for i, s := range samples {
				glog.V(1).Infof("Sample %d: %v", i, s)
				policy.AddSample(s)
			}

			policy.TrainNetwork()
		}()
	}

	wg.Wait()
	policy.AddCurrentExploiterToModel()
	if err := savePolicy(params, player, policy); err != nil {
		glog.Fatal(err)
	}
}

func loadPolicy(params RunParams) [2]*model.MCTSPSRO {
	lstm := model.NewLSTM(params.ModelParams)
	p0 := model.NewMCTSPSRO(lstm, params.SampleBufferSize, params.RetrainInterval, params.PredictionCacheSize)
	p1 := model.NewMCTSPSRO(lstm, params.SampleBufferSize, params.RetrainInterval, params.PredictionCacheSize)
	policies := [2]*model.MCTSPSRO{p0, p1}
	for player := range policies {
		filename := filepath.Join(params.ModelParams.OutputDir, fmt.Sprintf("player_%d.model", player))
		f, err := os.Open(filename)
		if err != nil {
			glog.Warningf("Unable to load player %d policy: %v", player, err)
			continue
		}
		defer f.Close()
		r := bufio.NewReader(f)

		policy, err := model.LoadMCTSPSRO(r)
		if err != nil {
			glog.Warningf("Unable to load player %d policy: %v", player, err)
			continue
		}

		policies[player] = policy
	}

	return policies
}

func savePolicy(params RunParams, player int, policy *model.MCTSPSRO) error {
	filename := filepath.Join(params.ModelParams.OutputDir, fmt.Sprintf("player_%d.model", player))
	glog.Infof("Saving player %d policy to: %v", player, filename)
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	return policy.SaveTo(w)
}

func playGame(game cfr.GameTreeNode, search *mcts.OneSidedISMCTS, opponentPolicy mcts.Policy, player int, params RunParams) []model.Sample {
	infoSet := game.(*alphacats.GameNode).GetInfoSet(gamestate.Player(player))
	beliefs := alphacats.NewBeliefState(opponentPolicy.GetPolicy, infoSet)

	var samples []model.Sample
	for game.Type() != cfr.TerminalNodeType {
		if game.Type() == cfr.ChanceNodeType {
			game, _ = game.SampleChild()
		} else if game.Player() != player { // Opponent.
			p := opponentPolicy.GetPolicy(game)
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
		} else {
			simulate(search, opponentPolicy, beliefs, params.NumMCTSIterations, params.MaxParallelSearches)
			is := game.InfoSet(game.Player()).(*alphacats.AbstractedInfoSet)
			p := search.GetPolicy(game)
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
			samples = append(samples, model.Sample{
				InfoSet: *is,
				Policy:  p,
			})
		}

		beliefs.Update(game.(*alphacats.GameNode).GetInfoSet(gamestate.Player(player)))
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

func simulate(search *mcts.OneSidedISMCTS, opponentPolicy mcts.Policy, beliefs *alphacats.BeliefState, n, nParallel int) {
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
				search.Run(rng, game, opponentPolicy)
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
