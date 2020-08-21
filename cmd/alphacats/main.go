// This version of alphacats uses one-sided IS-MCTS with a NN
// to guide search, in a PSRO framework.
package main

import (
	"bufio"
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
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

type RunParams struct {
	NumGamesPerEpoch    int
	MaxParallelGames    int
	NumMCTSIterations   int
	MaxParallelSearches int

	SamplingParams   SamplingParams
	Temperature      float64
	SampleBufferSize int
	MaxSampleReuse   int

	ModelParams model.Params
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
	flag.IntVar(&params.NumGamesPerEpoch, "games_per_epoch", 5000, "Number of games to play each epoch")
	flag.IntVar(&params.MaxParallelGames, "max_parallel_games", runtime.NumCPU(), "Number of games to run in parallel")
	flag.IntVar(&params.NumMCTSIterations, "search_iter", 2000, "Number of MCTS iterations to perform per move")
	flag.IntVar(&params.MaxParallelSearches, "max_parallel_searches", runtime.NumCPU(), "Number of searches per game to run in parallel")
	flag.IntVar(&params.SampleBufferSize, "sample_buffer_size", 200000, "Maximum number of training samples to keep")
	flag.IntVar(&params.MaxSampleReuse, "max_sample_reuse", 30, "Maximum number of times to reuse a sample.")
	flag.Float64Var(&params.Temperature, "temperature", 1.0,
		"Temperature used when selecting actions during play")
	flag.Int64Var(&params.SamplingParams.Seed, "sampling.seed", 123, "Random seed")
	flag.Float64Var(&params.SamplingParams.C, "sampling.c", 1.75,
		"Exploration factor C used in MCTS search")
	flag.StringVar(&params.ModelParams.OutputDir, "model.output_dir", "models",
		"Output directory for trained models")
	flag.IntVar(&params.ModelParams.NumEncodingWorkers, "model.encoding_workers", 4,
		"Maximum number of workers for training data encoding")
	flag.IntVar(&params.ModelParams.NumPredictionWorkers, "model.num_predict_workers", 4,
		"Number of background prediction workers")
	flag.IntVar(&params.ModelParams.MaxInferenceBatchSize, "model.predict_batch_size", 2048,
		"Maximum batch size for prediction")

	flag.Parse()

	rand.Seed(params.SamplingParams.Seed)
	go http.ListenAndServe("localhost:4123", nil)

	deck := cards.CoreDeck.AsSlice()
	cardsPerPlayer := 4
	lstm := model.NewLSTM(params.ModelParams)
	p0 := model.NewMCTSPSRO(lstm, params.SampleBufferSize, params.MaxSampleReuse)
	p1 := model.NewMCTSPSRO(lstm, params.SampleBufferSize, params.MaxSampleReuse)
	policies := []*model.MCTSPSRO{p0, p1}
	for epoch := 0; ; epoch++ {
		player := epoch % 2
		policy := policies[player]
		opponentPolicy := policies[1-player]
		search := mcts.NewOneSidedISMCTS(player, opponentPolicy, policy, float32(params.SamplingParams.C))
		glog.Infof("Starting epoch %d: Playing %d games to train approximate best response for player %d",
			epoch, params.NumGamesPerEpoch, player)
		var wg sync.WaitGroup
		sem := make(chan struct{}, params.MaxParallelGames)
		for i := 0; i < params.NumGamesPerEpoch; i++ {
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer func() {
					wg.Done()
					<-sem
				}()
				deal := alphacats.NewRandomDeal(deck, cardsPerPlayer)
				game := alphacats.NewGame(deal.DrawPile, deal.P0Deal, deal.P1Deal)
				infoSet := game.GetInfoSet(gamestate.Player(player))
				beliefs := alphacats.NewBeliefState(opponentPolicy.GetPolicy, infoSet)

				glog.Infof("Playing game with ~%d search iterations", params.NumMCTSIterations)
				samples := playGame(game, opponentPolicy, policy, search, beliefs, player, params)
				glog.Infof("Collected %d samples", len(samples))
				for _, s := range samples {
					policy.AddSample(s)
				}

				policy.TrainNetwork()
			}()
		}

		wg.Wait()
		policy.AddCurrentExploiterToModel()
	}
}

func playGame(game cfr.GameTreeNode, opponentPolicy, policy *model.MCTSPSRO, search *mcts.OneSidedISMCTS, beliefs *alphacats.BeliefState, player int, params RunParams) []model.Sample {
	var samples []model.Sample
	for game.Type() != cfr.TerminalNodeType {
		nodeType := game.Type()
		if nodeType == cfr.ChanceNodeType {
			game, _ = game.SampleChild()
		} else if game.Player() != player { // Opponent.
			p := opponentPolicy.GetPolicy(game)
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
		} else {
			simulate(search, beliefs, params.NumMCTSIterations, params.MaxParallelSearches)
			is := game.InfoSet(game.Player()).(*alphacats.AbstractedInfoSet)
			p := search.GetPolicy(game, float32(params.Temperature))
			selected := sampling.SampleOne(p, rand.Float32())
			game = game.GetChild(selected)
			samples = append(samples, model.Sample{
				InfoSet: *is,
				Policy:  p,
			})
		}

		beliefs.Update(nodeType, game.(*alphacats.GameNode).GetInfoSet(gamestate.Player(player)))
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

func simulate(search *mcts.OneSidedISMCTS, beliefs *alphacats.BeliefState, n, nParallel int) {
	var wg sync.WaitGroup
	nWorkers := min(n, nParallel) // TODO(palpant): Make this a flag.
	nPerWorker := n / nWorkers
	for worker := 0; worker < nWorkers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < nPerWorker; k++ {
				game := beliefs.SampleDeterminization()
				search.Run(game)
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
