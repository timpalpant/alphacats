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

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/model"
)

type cfrAlgo interface {
	Run(cfr.GameTreeNode) float32
}

func getCFRAlgo(policy cfr.StrategyProfile, samplingType string) cfrAlgo {
	switch samplingType {
	case "external":
		return cfr.NewExternalSampling(policy)
	case "outcome":
		return cfr.NewOutcomeSampling(policy, 0.1)
	case "chance":
		return cfr.NewChanceSampling(policy)
	default:
		panic(fmt.Errorf("unknown sampling type: %v", samplingType))
	}
}

func newPolicy(cfrType string, params model.Params, bufSize int) cfr.StrategyProfile {
	switch cfrType {
	case "tabular":
		return cfr.NewPolicyTable(cfr.DiscountParams{})
	case "deep":
		lstm := model.NewLSTM(params)
		buffers := []deepcfr.Buffer{
			deepcfr.NewThreadSafeReservoirBuffer(bufSize),
			deepcfr.NewThreadSafeReservoirBuffer(bufSize),
		}
		return deepcfr.New(lstm, buffers)
	default:
		panic(fmt.Errorf("unknown CFR type: %v", cfrType))
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

func main() {
	params := model.Params{}
	deckType := flag.String("decktype", "test", "Type of deck to use (core, test)")
	cfrType := flag.String("cfrtype", "tabular", "Type of CFR to run (tabular, deep)")
	samplingType := flag.String("sampling", "external",
		"Type of sampling to perform (external, chance, outcome, average)")
	seed := flag.Int64("seed", 123, "Random seed")
	iter := flag.Int("iter", 100, "Number of DeepCFR iterations to perform")
	bufSize := flag.Int("buf_size", 10000000, "Size of reservoir sample buffer")
	traversalsPerIter := flag.Int("traversals_per_iter", 30000,
		"Number of ES-CFR traversals to perform each iteration")
	outputDir := flag.String("output_dir", "", "Directory to save policies to")
	resume := flag.String("resume", "", "Resume training with given model")
	numSamplingThreads := flag.Int("num_sampling_threads", 256,
		"Max number of sampling runs to perform in parallel")
	flag.IntVar(&params.BatchSize, "batch_size", 4096,
		"Size of minibatches to save for network training")
	flag.StringVar(&params.ModelOutputDir, "model_dir", "",
		"Directory to save trained network models to")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	if err := os.MkdirAll(*outputDir, 0777); err != nil {
		glog.Fatal(err)
	}

	var policy cfr.StrategyProfile
	if *resume == "" {
		policy = newPolicy(*cfrType, params, *bufSize)
	} else {
		policy = loadPolicy(*cfrType, *resume)
	}
	opt := getCFRAlgo(policy, *samplingType)
	deck, cardsPerPlayer := getDeck(*deckType)

	var wg sync.WaitGroup
	// Becuse we are generally rate-limited by the speed at which we can make
	// model predictions, and because the GPU can perform batches of predictions
	// more efficiently (N predictions in < N x 1 sample time), we want to have
	// a bunch of collection runs going in parallel, so that when we make a
	// prediction it is usually for a larger batch of samples.
	//
	// Our GPU (NVIDIA 1060) seems to top out at a concurrency of ~1024,
	// see benchmarks in model/lstm_test.go.
	sem := make(chan struct{}, *numSamplingThreads)
	for t := policy.Iter(); t <= *iter; t++ {
		glog.V(1).Infof("[t=%d] Collecting %d samples with %d threads",
			t, *traversalsPerIter, cap(sem))
		start := time.Now()
		for k := 1; k <= *traversalsPerIter; k++ {
			glog.V(3).Infof("[k=%d] Running CFR iteration on random game", k)
			game := alphacats.NewRandomGame(deck, cardsPerPlayer)
			sem <- struct{}{}
			wg.Add(1)
			go func() {
				opt.Run(game)
				<-sem
				wg.Done()
			}()
		}
		wg.Wait()
		glog.V(1).Infof("[t=%d] Finished collecting samples (took: %v)", t, time.Since(start))

		glog.V(1).Infof("[t=%d] Training network", t)
		start = time.Now()
		policy.Update()
		glog.V(1).Infof("[t=%d] Finished training network (took: %v)", t, time.Since(start))

		// Save 10 snapshots throughout the course of training.
		if t%(*iter/10) == 0 {
			if err := savePolicy(policy, *outputDir, t); err != nil {
				glog.Fatal(err)
			}
		}
	}
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

func loadPolicy(cfrType, filename string) cfr.StrategyProfile {
	glog.Infof("Loading saved %v policy from: %v", cfrType, filename)
	start := time.Now()
	f, err := os.Open(filename)
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
