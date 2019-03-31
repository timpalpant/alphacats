package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/golang/glog"
	gzip "github.com/klauspost/pgzip"
	"github.com/syndtr/goleveldb/leveldb/filter"
	"github.com/syndtr/goleveldb/leveldb/opt"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"
	"github.com/timpalpant/go-cfr/ldbstore"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/model"
)

type RunParams struct {
	DeckType         string
	CFRType          string
	NumCFRIterations int

	SamplingParams SamplingParams
	DeepCFRParams  DeepCFRParams
	LDBParams      LevelDBParams

	OutputDir  string
	ResumeFrom string
}

type SamplingParams struct {
	SamplingType       string
	SampledActionsType string
	LDBParams          LevelDBParams
	NumSamplingThreads int
	Seed               int64
}

type DeepCFRParams struct {
	BufferType        string
	BufferSize        int
	LDBParams         LevelDBParams
	TraversalsPerIter int
	ModelParams       model.Params
}

type LevelDBParams struct {
	BlockCacheCapacity int
	WriteBuffer        int
	BloomFilterNumBits int
}

func (p LevelDBParams) ToOpts() *opt.Options {
	o := &opt.Options{
		BlockCacheCapacity: p.BlockCacheCapacity,
		WriteBuffer:        p.WriteBuffer,
		NoSync:             true,
	}

	if p.BloomFilterNumBits > 0 {
		o.Filter = filter.NewBloomFilter(p.BloomFilterNumBits)
	}

	return o
}

type cfrAlgo interface {
	Run(cfr.GameTreeNode) float32
}

func getSampledActionsFactory(params RunParams) cfr.SampledActionsFactory {
	switch params.SamplingParams.SampledActionsType {
	case "leveldb":
		return func() cfr.SampledActions {
			tmpDir, err := ioutil.TempDir(params.OutputDir, "sampled-actions-")
			if err != nil {
				glog.Fatal(err)
			}

			opts := params.SamplingParams.LDBParams.ToOpts()
			ss, err := ldbstore.NewLDBSampledActions(tmpDir, opts)
			if err != nil {
				glog.Fatal(err)
			}

			return ss
		}
	default:
		return cfr.NewSampledActionsMap
	}
}

func getCFRAlgo(params RunParams, policy cfr.StrategyProfile) cfrAlgo {
	switch params.SamplingParams.SamplingType {
	case "external":
		sampledActionsFactory := getSampledActionsFactory(params)
		return cfr.NewExternalSampling(policy, sampledActionsFactory)
	case "outcome":
		return cfr.NewOutcomeSampling(policy, 0.1)
	case "chance":
		return cfr.NewChanceSampling(policy)
	default:
		panic(fmt.Errorf("unsupported sampling type: %v", params.SamplingParams.SamplingType))
	}
}

func getReservoirBuffer(params DeepCFRParams) deepcfr.Buffer {
	switch params.BufferType {
	case "leveldb":
		t := time.Now().UnixNano()
		bufPath := filepath.Join(params.ModelParams.OutputDir, fmt.Sprintf("buffer1-%d", t))
		opts := params.LDBParams.ToOpts()
		buf, err := ldbstore.NewReservoirBuffer(bufPath, opts, params.BufferSize)
		if err != nil {
			glog.Fatal(err)
		}
		return buf
	default:
		return deepcfr.NewThreadSafeReservoirBuffer(params.BufferSize)
	}
}

func newPolicy(params RunParams) cfr.StrategyProfile {
	switch params.CFRType {
	case "tabular":
		return cfr.NewPolicyTable(cfr.DiscountParams{})
	case "leveldb":
		opts := params.LDBParams.ToOpts()
		policy, err := ldbstore.New(params.OutputDir, opts, cfr.DiscountParams{})
		if err != nil {
			glog.Fatal(err)
		}
		return policy
	case "deep":
		dCFRParams := params.DeepCFRParams
		dCFRParams.ModelParams.OutputDir = filepath.Join(params.OutputDir, "models")
		lstm := model.NewLSTM(dCFRParams.ModelParams)
		buffers := []deepcfr.Buffer{
			getReservoirBuffer(dCFRParams),
			getReservoirBuffer(dCFRParams),
		}
		return deepcfr.New(lstm, buffers)
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

func main() {
	var params RunParams
	flag.StringVar(&params.DeckType, "decktype", "test", "Type of deck to use (core, test)")
	flag.StringVar(&params.CFRType, "cfrtype", "tabular", "Type of CFR to run (tabular, deep)")
	flag.IntVar(&params.NumCFRIterations, "iter", 100, "Number of DeepCFR iterations to perform")
	flag.StringVar(&params.SamplingParams.SamplingType, "sampling", "external",
		"Type of sampling to perform (external, chance, outcome, average)")
	flag.IntVar(&params.SamplingParams.NumSamplingThreads, "num_sampling_threads", 256,
		"Max number of sampling runs to perform in parallel")
	flag.Int64Var(&params.SamplingParams.Seed, "seed", 123, "Random seed")
	flag.IntVar(&params.DeepCFRParams.BufferSize, "buf_size", 10000000, "Size of reservoir sample buffer")
	flag.IntVar(&params.DeepCFRParams.TraversalsPerIter, "traversals_per_iter", 30000,
		"Number of ES-CFR traversals to perform each iteration")
	flag.IntVar(&params.DeepCFRParams.ModelParams.BatchSize, "batch_size", 4096,
		"Size of minibatches to save for network training")
	flag.StringVar(&params.OutputDir, "output_dir", "", "Directory to save policies to")
	flag.StringVar(&params.ResumeFrom, "resume", "", "Resume training with given model")
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
	opt := getCFRAlgo(params, policy)
	deck, cardsPerPlayer := getDeck(params.DeckType)

	var wg sync.WaitGroup
	// Becuse we are generally rate-limited by the speed at which we can make
	// model predictions, and because the GPU can perform batches of predictions
	// more efficiently (N predictions in < N x 1 sample time), we want to have
	// a bunch of collection runs going in parallel, so that when we make a
	// prediction it is usually for a larger batch of samples.
	//
	// Our GPU (NVIDIA 1060) seems to top out at a concurrency of ~1024,
	// see benchmarks in model/lstm_test.go.
	sem := make(chan struct{}, params.SamplingParams.NumSamplingThreads)
	for t := policy.Iter(); t <= params.NumCFRIterations; t++ {
		glog.V(1).Infof("[t=%d] Collecting %d samples with %d threads",
			t, params.DeepCFRParams.TraversalsPerIter, cap(sem))
		start := time.Now()
		for k := 1; k <= params.DeepCFRParams.TraversalsPerIter; k++ {
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
