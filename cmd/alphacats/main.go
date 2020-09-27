// This version of alphacats uses one-sided IS-MCTS with a NN
// to guide search, in a PSRO framework.
package main

import (
	"bufio"
	"encoding/gob"
	"expvar"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/mcts"
	"github.com/timpalpant/go-cfr/sampling"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
	"github.com/timpalpant/alphacats/matrixgame"
	"github.com/timpalpant/alphacats/model"
)

var stdin = bufio.NewReader(os.Stdin)
var start = time.Now()

var (
	gamesRemaining    = expvar.NewInt("games_remaining")
	gamesPlayed       = expvar.NewInt("games_played")
	gamesInFlight     = expvar.NewInt("games_in_flight")
	numSamples        = expvar.NewInt("num_samples")
	p0p0Wins          = expvar.NewInt("num_wins/player0_train/player0")
	p0p1Wins          = expvar.NewInt("num_wins/player0_train/player1")
	p1p0Wins          = expvar.NewInt("num_wins/player1_train/player0")
	p1p1Wins          = expvar.NewInt("num_wins/player1_train/player1")
	searchesPerformed = expvar.NewInt("searches_performed")
	searchesInFlight  = expvar.NewInt("searches_in_flight")
	searchesPerSecond = expvar.NewFloat("searches_per_sec")
)

type RunParams struct {
	Deck           []cards.Card
	CardsPerPlayer int

	BootstrapSamplesDir        string
	MaxNumGamesPerEpoch        int
	EarlyStoppingPatience      int
	MaxParallelGames           int
	NumMCTSIterationsExpensive int
	NumMCTSIterationsCheap     int
	ExpensiveMoveFraction      float64
	MaxParallelSearches        int
	NumMetaPolicySimulations   int
	NumMetaPolicyIterations    int

	SamplingParams   SamplingParams
	Temperature      float64
	SampleBufferSize int
	RetrainInterval  int

	ModelParams         model.Params
	PredictionCacheSize int
}

type SamplingParams struct {
	Seed int64
	C    float64
}

func main() {
	params := RunParams{
		Deck:           cards.CoreDeck.AsSlice(),
		CardsPerPlayer: 4,
	}
	flag.StringVar(&params.BootstrapSamplesDir, "bootstrap_samples_dir", "models/bootstrap-training-data",
		"Directory with bootstrap training data for initial model")
	flag.IntVar(&params.MaxNumGamesPerEpoch, "games_per_epoch", 25000,
		"Maximum number of games to play each epoch")
	flag.IntVar(&params.EarlyStoppingPatience, "early_stopping_patience", 3,
		"Stop epoch early if no improvement for this many trainings")
	flag.IntVar(&params.MaxParallelGames, "max_parallel_games", 768,
		"Number of games to run in parallel")
	flag.IntVar(&params.NumMCTSIterationsExpensive, "search_iter_expensive", 800,
		"Number of MCTS iterations to perform on expensive moves")
	flag.IntVar(&params.NumMCTSIterationsCheap, "search_iter_cheap", 200,
		"Number of MCTS iterations to perform on cheap moves")
	flag.Float64Var(&params.ExpensiveMoveFraction, "expensive_fraction", 0.5,
		"Fraction of moves to perform expensive search")
	flag.IntVar(&params.MaxParallelSearches, "max_parallel_searches", 64,
		"Number of searches per game to run in parallel")
	flag.IntVar(&params.SampleBufferSize, "sample_buffer_size", 500000,
		"Maximum number of training samples to keep")
	flag.IntVar(&params.RetrainInterval, "retrain_interval", 10000,
		"How many samples to collect before retraining.")
	flag.IntVar(&params.NumMetaPolicySimulations, "num_meta_policy_simulations", 10000,
		"How many games to simulate when estimating meta policy distribution")
	flag.IntVar(&params.NumMetaPolicyIterations, "num_meta_policy_iterations", 1000000,
		"Number of iterations of fictitious play to perform when estimating meta policy distribution")
	flag.Float64Var(&params.Temperature, "temperature", 0.8,
		"Temperature used when selecting actions during play")
	flag.Int64Var(&params.SamplingParams.Seed, "sampling.seed", 123, "Random seed")
	flag.Float64Var(&params.SamplingParams.C, "sampling.c", 1.2,
		"Exploration factor C used in MCTS search")
	flag.StringVar(&params.ModelParams.OutputDir, "model.output_dir", "models",
		"Output directory for trained models")
	flag.IntVar(&params.ModelParams.NumEncodingWorkers, "model.encoding_workers", 1,
		"Maximum number of workers for training data encoding")
	flag.IntVar(&params.ModelParams.NumPredictionWorkers, "model.num_predict_workers", 1,
		"Number of background prediction workers")
	flag.IntVar(&params.ModelParams.MaxInferenceBatchSize, "model.predict_batch_size", 16384,
		"Maximum batch size for prediction")
	flag.IntVar(&params.PredictionCacheSize, "prediction_cache_size", 100000,
		"Size of LRU prediction cache per model")

	flag.Parse()

	rand.Seed(params.SamplingParams.Seed)
	go http.ListenAndServe("localhost:4123", nil)

	// Initialize policies. Bootstrap from training data if available.
	policies := loadPolicy(params)
	var wg sync.WaitGroup
	if policies[0].Len() == 0 {
		wg.Add(1)
		go func() {
			bootstrap(policies[0], 0, params)
			wg.Done()
		}()
	}
	if policies[1].Len() == 0 {
		wg.Add(1)
		go func() {
			bootstrap(policies[1], 1, params)
			wg.Done()
		}()
	}
	wg.Wait()

	// Run PSRO: Each epoch, train an approximate best response to the opponent's
	// current policy distribution. Then add the new policies to the meta-model.
	start = time.Now()
	var winRateMatrix [][]float64
	for epoch := policies[0].Len(); ; epoch++ {
		winRateMatrix = updateNashWeights(policies[0], policies[1], winRateMatrix, params)
		for player := 0; player < 1; player++ {
			if err := savePolicy(params, player, policies[player], epoch-1, -1); err != nil {
				glog.Fatal(err)
			}
		}

		glog.Infof("Starting epoch %d: Playing %d games to train approximate best responses",
			epoch, params.MaxNumGamesPerEpoch)
		wg.Add(2)
		go func() {
			runEpoch(policies, 0, params, epoch)
			wg.Done()
		}()
		// NB: Work around some CUDA initialization race that leads to segfault.
		time.Sleep(20 * time.Second)
		go func() {
			runEpoch(policies, 1, params, epoch)
			wg.Done()
		}()
		wg.Wait()

		// Update meta-model with new best response policies.
		for player := 0; player < 1; player++ {
			policies[player].AddCurrentExploiterToModel()
		}
	}
}

// Given our collection of policies (models) for player 0 and player 1,
// determine how often they should play each model based on empirical pairwise
// win rates of all models against each other.
func updateNashWeights(policy0, policy1 *model.MCTSPSRO, winRateMatrix [][]float64, params RunParams) [][]float64 {
	p0Policies := policy0.GetPolicies()
	p1Policies := policy1.GetPolicies()
	glog.Infof("Updating win rate matrix with %d x %d policies",
		len(p0Policies), len(p1Policies))
	winRateMatrix = updateWinRateMatrix(p0Policies, p1Policies, winRateMatrix, params)
	glog.Infof("Solving meta game with %d iterations of fictitious play",
		params.NumMetaPolicyIterations)
	p0Weights, p1Weights := matrixgame.FictitiousPlay(winRateMatrix, params.NumMetaPolicyIterations)
	policy0.AssignWeights(p0Weights)
	policy1.AssignWeights(p1Weights)
	return winRateMatrix
}

func updateWinRateMatrix(p0Policies, p1Policies []mcts.Policy, winRateMatrix [][]float64, params RunParams) [][]float64 {
	// Add rows and columns to the matrix for any new policies. Initialize to NaN.
	for i := len(winRateMatrix); i < len(p0Policies); i++ {
		winRateMatrix = append(winRateMatrix, makeNaNs(len(p1Policies)))
	}
	for i, winRate := range winRateMatrix {
		winRateMatrix[i] = append(winRate, makeNaNs(len(p1Policies)-len(winRate))...)
	}

	// Perform simulations to populate all NaN entries.
	for i := range winRateMatrix {
		for j := range winRateMatrix[i] {
			if !math.IsNaN(winRateMatrix[i][j]) {
				continue // Already populated.
			}

			glog.Infof("Simulating %d games between policies %d, %d",
				params.NumMetaPolicySimulations, i, j)
			winRate := battlePolicies(p0Policies[i], p1Policies[j], params)
			winRateMatrix[i][j] = winRate
		}
	}

	glog.Infof("Win rate matrix is now:")
	for _, row := range winRateMatrix {
		glog.Infof("\t%v", row)
	}
	return winRateMatrix
}

func makeNaNs(n int) []float64 {
	result := make([]float64, n)
	for i := range result {
		result[i] = math.NaN()
	}
	return result
}

func battlePolicies(p0Policy, p1Policy mcts.Policy, params RunParams) float64 {
	wins := 0
	var mx sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, params.MaxParallelGames*params.MaxParallelSearches)
	for i := 1; i <= params.NumMetaPolicySimulations; i++ {
		if i == 2 {
			// NB: Work around some CUDA initialization race that leads to segfault.
			time.Sleep(15 * time.Second)
		} else if i%1000 == 0 {
			glog.Infof("...simulated %d games", i)
		}

		sem <- struct{}{}
		wg.Add(1)
		go func() {
			defer func() {
				<-sem
				wg.Done()
			}()
			winner := playOneVersusMatch(p0Policy, p1Policy, params)
			if winner == 0 {
				mx.Lock()
				defer mx.Unlock()
				wins++
			}
		}()
	}

	wg.Wait()
	return float64(wins) / float64(params.NumMetaPolicySimulations)
}

func playOneVersusMatch(p0Policy, p1Policy mcts.Policy, params RunParams) (winner int) {
	// NB: NewRandomDeal shuffles the passed deck.
	deck := make([]cards.Card, len(params.Deck))
	copy(deck, params.Deck)
	deal := alphacats.NewRandomDeal(deck, params.CardsPerPlayer)
	var game cfr.GameTreeNode = alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
	for game.Type() != cfr.TerminalNodeType {
		if game.Type() == cfr.ChanceNodeType {
			game, _ = game.SampleChild()
		} else {
			policy := p0Policy
			if game.Player() == 1 {
				policy = p1Policy
			}

			p := policy.GetPolicy(game)
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
		}

	}

	return game.Player()
}

func glob(dir string, prefix, ext string) ([]string, error) {
	var files []string
	err := filepath.Walk(dir, func(path string, f os.FileInfo, err error) error {
		if strings.HasPrefix(filepath.Base(path), prefix) && filepath.Ext(path) == ext {
			files = append(files, path)
		}
		return nil
	})

	return files, err
}

func bootstrap(policy *model.MCTSPSRO, player int, params RunParams) {
	trainingData, err := glob(params.BootstrapSamplesDir, fmt.Sprintf("player_%d", player), ".samples")
	if err != nil || len(trainingData) > 0 {
		glog.Infof("Training initial player %d model with bootstrap data from %d files",
			player, len(trainingData))
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

func runEpoch(policies [2]*model.MCTSPSRO, player int, params RunParams, epoch int) {
	gamesRemaining.Add(int64(params.MaxNumGamesPerEpoch))

	var mx sync.Mutex
	var wg sync.WaitGroup
	sem := make(chan struct{}, params.MaxParallelGames)
	policy := policies[player]
	opponent := policies[1-player]
	numSamples.Set(0)
	wins, losses := p0p0Wins, p0p1Wins
	if player == 1 {
		wins, losses = p1p1Wins, p1p0Wins
	}
	var winRates []float64
	var prevWins, prevLosses int64
	numSamplesSinceLastTrain := 0
	modelIter := 0
	for i := 0; i < params.MaxNumGamesPerEpoch && !shouldStopEarly(winRates, params); i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer func() {
				gamesRemaining.Add(-1)
				wg.Done()
				<-sem
			}()
			// NB: NewRandomDeal shuffles the passed deck.
			deck := make([]cards.Card, len(params.Deck))
			copy(deck, params.Deck)
			deal := alphacats.NewRandomDeal(deck, params.CardsPerPlayer)
			game := alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
			opponentPolicy := opponent.SamplePolicy()
			ismcts := mcts.NewOneSidedISMCTS(player, policy,
				float32(params.SamplingParams.C), float32(params.Temperature))
			glog.Infof("Playing game with %d/%d search iterations",
				params.NumMCTSIterationsExpensive,
				params.NumMCTSIterationsCheap)
			samples := playGame(game, ismcts, opponentPolicy, player, params)
			glog.Infof("Collected %d samples", len(samples))
			gamesPlayed.Add(1)
			numSamples.Add(int64(len(samples)))

			for i, s := range samples {
				glog.V(1).Infof("Sample %d: %v", i, s)
				policy.AddSample(s)
			}

			// NB: When we are player 1, it is possible that player 0 lost on the first
			// turn before we had any chance to play.
			if len(samples) == 0 || samples[len(samples)-1].Value != 1.0 {
				losses.Add(1)
			} else {
				wins.Add(1)
			}

			mx.Lock()
			defer mx.Unlock()
			numSamplesSinceLastTrain += len(samples)
		}()

		mx.Lock()
		needsRetrain := (numSamplesSinceLastTrain >= params.RetrainInterval)
		if needsRetrain {
			newWins := wins.Value() - prevWins
			newLosses := losses.Value() - prevLosses
			winRate := float64(newWins) / float64(newWins+newLosses)
			winRates = append(winRates, winRate)
			glog.Infof("Win rates: %v", winRates)
			prevWins = wins.Value()
			prevLosses = losses.Value()

			policy.TrainNetwork()
			numSamplesSinceLastTrain = 0
			modelIter++
			if err := savePolicy(params, player, policy, epoch, modelIter); err != nil {
				glog.Fatal(err)
			}

		}
		mx.Unlock()
	}

	wg.Wait()
}

func shouldStopEarly(winRates []float64, params RunParams) bool {
	_, idx := argMax(winRates)
	return idx < len(winRates)-1-params.EarlyStoppingPatience
}

func argMax(vs []float64) (float64, int) {
	best := -math.MaxFloat64
	bestIdx := 0
	for i, v := range vs {
		if v > best {
			best = v
			bestIdx = i
		}
	}

	return best, bestIdx
}

func loadPolicy(params RunParams) [2]*model.MCTSPSRO {
	p0Params := params.ModelParams
	p0Params.OutputDir = filepath.Join(p0Params.OutputDir, "player0")
	lstm0 := model.NewLSTM(p0Params)
	p0 := model.NewMCTSPSRO(lstm0, params.SampleBufferSize, params.PredictionCacheSize)

	p1Params := params.ModelParams
	p1Params.OutputDir = filepath.Join(p1Params.OutputDir, "player1")
	lstm1 := model.NewLSTM(p1Params)
	p1 := model.NewMCTSPSRO(lstm1, params.SampleBufferSize, params.PredictionCacheSize)

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

func savePolicy(params RunParams, player int, policy *model.MCTSPSRO, epoch, modelIter int) error {
	filename := filepath.Join(params.ModelParams.OutputDir, fmt.Sprintf("player_%d.model", player))
	if err := savePolicyToFile(policy, filename); err != nil {
		return err
	}

	filename = filepath.Join(params.ModelParams.OutputDir,
		fmt.Sprintf("player_%d.model.%9d.epoch_%04d.iter_%04d",
			player, start.Nanosecond(), epoch, modelIter))
	return savePolicyToFile(policy, filename)
}

func savePolicyToFile(policy *model.MCTSPSRO, filename string) error {
	glog.Infof("Saving policy to: %v", filename)
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	if err := policy.SaveTo(w); err != nil {
		_ = f.Close()
		return err
	}
	if err := w.Flush(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

func playGame(game cfr.GameTreeNode, search *mcts.OneSidedISMCTS, opponentPolicy mcts.Policy, player int, params RunParams) []model.Sample {
	gamesInFlight.Add(1)
	defer gamesInFlight.Add(-1)
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
			numMCTSIterations := params.NumMCTSIterationsCheap
			expensiveSearch := (rand.Float64() < params.ExpensiveMoveFraction)
			if expensiveSearch {
				numMCTSIterations = params.NumMCTSIterationsExpensive
			}
			simulate(search, opponentPolicy, beliefs, numMCTSIterations, params.MaxParallelSearches)
			is := game.InfoSet(game.Player()).(*alphacats.AbstractedInfoSet)
			p := search.GetPolicy(game)
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
			if expensiveSearch {
				samples = append(samples, model.Sample{
					InfoSet: *is,
					Policy:  p,
				})
			}
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
				searchesInFlight.Add(1)
				search.Run(rng, game, opponentPolicy)
				searchesInFlight.Add(-1)
				searchesPerformed.Add(1)
				searchesPerSecond.Set(float64(searchesPerformed.Value()) / time.Since(start).Seconds())
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
