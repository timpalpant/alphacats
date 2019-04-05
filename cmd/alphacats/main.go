package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/golang/glog"
	gzip "github.com/klauspost/pgzip"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"
	"github.com/timpalpant/go-cfr/sampling"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/model"
)

const (
	KiB = 1024
	MiB = 1024 * KiB
	GiB = 1024 * MiB
)

type RunParams struct {
	DeckType         string
	CFRType          string
	NumCFRIterations int

	SamplingParams SamplingParams
	DeepCFRParams  DeepCFRParams

	OutputDir  string
	ResumeFrom string
}

type SamplingParams struct {
	RobustSamplingK    int
	NumSamplingThreads int
	Seed               int64
}

type DeepCFRParams struct {
	BufferType        string
	BufferSize        int
	TraversalsPerIter int
	ModelParams       model.Params
}

func newPolicy(params RunParams) cfr.StrategyProfile {
	switch params.CFRType {
	case "tabular":
		return cfr.NewPolicyTable(cfr.DiscountParams{})
	case "deep":
		dCFRParams := params.DeepCFRParams
		dCFRParams.ModelParams.OutputDir = filepath.Join(params.OutputDir, "models")
		if err := os.MkdirAll(dCFRParams.ModelParams.OutputDir, 0777); err != nil {
			glog.Fatal(err)
		}

		lstm := model.NewLSTM(dCFRParams.ModelParams)
		buffers := []deepcfr.Buffer{
			deepcfr.NewReservoirBuffer(dCFRParams.BufferSize, params.SamplingParams.NumSamplingThreads),
			deepcfr.NewReservoirBuffer(dCFRParams.BufferSize, params.SamplingParams.NumSamplingThreads),
		}
		return deepcfr.New(lstm, buffers, true)
	default:
		panic(fmt.Errorf("unknown CFR type: %v", params.CFRType))
	}
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

func collectSamples(policy cfr.StrategyProfile, params RunParams) {
	deck, cardsPerPlayer := getDeck(params.DeckType)
	sem := make(chan struct{}, params.SamplingParams.NumSamplingThreads)
	var wg sync.WaitGroup
	for k := 1; k <= params.DeepCFRParams.TraversalsPerIter; k++ {
		sem <- struct{}{}
		glog.V(2).Infof("[k=%d] Running CFR iteration on random game", k)
		wg.Add(1)
		go func() {
			game := alphacats.NewRandomGame(deck, cardsPerPlayer)
			sampler := sampling.NewRobustSampler(params.SamplingParams.RobustSamplingK)
			walker := cfr.NewGeneralizedSampling(policy, cfr.SamplingParams{
				Sampler:               sampler,
				ProbeUnsampledActions: true,
			})
			walker.Run(game)
			<-sem
			wg.Done()
		}()
	}

	wg.Wait()
}

func main() {
	var params RunParams
	flag.StringVar(&params.DeckType, "decktype", "test", "Type of deck to use (core, test)")
	flag.StringVar(&params.CFRType, "cfrtype", "tabular", "Type of CFR to run (tabular, deep)")
	flag.IntVar(&params.NumCFRIterations, "iter", 100, "Number of DeepCFR iterations to perform")
	flag.StringVar(&params.OutputDir, "output_dir", "", "Directory to save policies to")
	flag.StringVar(&params.ResumeFrom, "resume", "", "Resume training with given model")

	flag.IntVar(&params.SamplingParams.NumSamplingThreads, "sampling.num_sampling_threads", 256,
		"Max number of sampling runs to perform in parallel")
	flag.IntVar(&params.SamplingParams.RobustSamplingK, "sampling.max_num_actions", 3,
		"Max number of actions to sample for traversing player in robust sampling")
	flag.Int64Var(&params.SamplingParams.Seed, "sampling.seed", 123, "Random seed")

	flag.IntVar(&params.DeepCFRParams.BufferSize,
		"deepcfr.buffer.size", 10000000,
		"Number of samples to keep in reservoir sample buffer")
	flag.IntVar(&params.DeepCFRParams.TraversalsPerIter,
		"deepcfr.traversals_per_iter", 1,
		"Number of ES-CFR traversals to perform each iteration")
	flag.IntVar(&params.DeepCFRParams.ModelParams.BatchSize,
		"deepcfr.model.batch_size", 2000,
		"Size of minibatches to save for network training")
	flag.IntVar(&params.DeepCFRParams.ModelParams.NumEncodingWorkers,
		"deepcfr.model.num_encoding_workers", 4,
		"Number of worker threads for prediction feature encoding")
	flag.IntVar(&params.DeepCFRParams.ModelParams.MaxTrainingDataWorkers,
		"deepcfr.model.max_training_data_workers", 24,
		"Number of worker threads for training data encoding")
	flag.Parse()

	rand.Seed(params.SamplingParams.Seed)
	go http.ListenAndServe("localhost:4123", nil)

	if err := os.MkdirAll(params.OutputDir, 0777); err != nil {
		glog.Fatal(err)
	}

	var policy cfr.StrategyProfile
	if params.ResumeFrom == "" {
		policy = newPolicy(params)
	} else {
		policy = loadPolicy(params)
	}

	// Becuse we are generally rate-limited by the speed at which we can make
	// model predictions, and because the GPU can perform batches of predictions
	// more efficiently (N predictions in < N x 1 sample time), we want to have
	// a bunch of collection runs going in parallel, so that when we make a
	// prediction it is usually for a larger batch of samples.
	//
	// Our GPU (NVIDIA 1060) seems to top out at a concurrency of ~1024,
	// see benchmarks in model/lstm_test.go.
	for t := policy.Iter(); t <= params.NumCFRIterations; t++ {
		glog.V(1).Infof("[t=%d] Collecting %d samples with %d threads",
			t, params.DeepCFRParams.TraversalsPerIter, params.SamplingParams.NumSamplingThreads)
		start := time.Now()
		collectSamples(policy, params)
		glog.V(1).Infof("[t=%d] Finished collecting samples (took: %v)", t, time.Since(start))

		glog.V(1).Infof("[t=%d] Training network", t)
		start = time.Now()
		policy.Update()
		glog.V(1).Infof("[t=%d] Finished training network (took: %v)", t, time.Since(start))

		// Save 10 snapshots throughout the course of training.
		if shouldSave(t, params.NumCFRIterations) {
			if err := savePolicy(policy, params.OutputDir, t); err != nil {
				glog.Fatal(err)
			}
		}
	}
}

func shouldSave(t, numIter int) bool {
	return t%(numIter/10) == 0
}

func savePolicy(policy cfr.StrategyProfile, outputDir string, iter int) error {
	name := fmt.Sprintf("iter_%d.policy", iter)
	outputFile := filepath.Join(outputDir, name)
	glog.Infof("[t=%d] Saving current policy to %v", iter, outputFile)
	f, err := os.Create(outputFile)
	if err != nil {
		glog.Fatal(err)
	}
	defer f.Close()

	w := gzip.NewWriter(f)
	defer w.Close()

	enc := gob.NewEncoder(w)
	// Need to pass pointer to interface so that Gob sees the interface rather
	// than the concrete type. See the example in encoding/gob.
	return enc.Encode(&policy)
}

func loadPolicy(params RunParams) cfr.StrategyProfile {
	glog.Infof("Loading saved policy from: %v", params.ResumeFrom)
	start := time.Now()
	f, err := os.Open(params.ResumeFrom)
	if err != nil {
		glog.Fatal(err)
	}
	defer f.Close()

	r, err := gzip.NewReader(f)
	if err != nil {
		glog.Fatal(err)
	}
	defer r.Close()

	var policy cfr.StrategyProfile
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&policy); err != nil {
		glog.Fatal(err)
	}

	glog.Infof("Finished loading policy (took: %v)", time.Since(start))
	return policy
}
