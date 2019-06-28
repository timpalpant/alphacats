package main

import (
	"expvar"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/golang/glog"
	gzip "github.com/klauspost/pgzip"
	rocksdb "github.com/tecbot/gorocksdb"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"
	"github.com/timpalpant/go-cfr/rdbstore"
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

var (
	gamesInProgress = expvar.NewInt("games_in_progress")
	gamesRemaining  = expvar.NewInt("games_remaining")
)

type RunParams struct {
	DeckType         string
	CFRType          string
	NumCFRIterations int

	SamplingParams SamplingParams
	DeepCFRParams  DeepCFRParams
	RDBParams      RocksDBParams

	OutputDir  string
	ResumeFrom string
}

type SamplingParams struct {
	MaxNumActionsK     int
	ExplorationEps     float64
	NumSamplingThreads int
	Seed               int64
}

type RocksDBParams struct {
	Path               string
	BlockCacheCapacity int
	WriteBuffer        int
	BloomFilterNumBits int
}

func (p RocksDBParams) Render(path string) rdbstore.Params {
	params := rdbstore.DefaultParams(path)

	params.Options.IncreaseParallelism(runtime.NumCPU())
	params.Options.SetCompression(rocksdb.LZ4Compression)
	params.Options.SetCreateIfMissing(true)
	params.Options.SetUseFsync(false)
	params.Options.SetWriteBufferSize(p.WriteBuffer)

	bOpts := rocksdb.NewDefaultBlockBasedTableOptions()
	bOpts.SetBlockSize(32 * 1024)
	blockCache := rocksdb.NewLRUCache(p.BlockCacheCapacity)
	bOpts.SetBlockCache(blockCache)
	if p.BloomFilterNumBits > 0 {
		filter := rocksdb.NewBloomFilter(p.BloomFilterNumBits)
		bOpts.SetFilterPolicy(filter)
	}
	params.Options.SetBlockBasedTableFactory(bOpts)

	params.WriteOptions.DisableWAL(true)
	params.WriteOptions.SetSync(false)

	return params
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
		return cfr.NewPolicyTable(cfr.DiscountParams{
			LinearWeighting:       true,
			UseRegretMatchingPlus: true,
		})
	case "rocksdb":
		opts := params.RDBParams.Render(params.OutputDir)
		policy, err := rdbstore.New(opts, cfr.DiscountParams{
			LinearWeighting:       true,
			UseRegretMatchingPlus: true,
		})

		if err != nil {
			glog.Fatal(err)
		}

		return policy
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
		baselineBuffers := []deepcfr.Buffer{
			deepcfr.NewReservoirBuffer(dCFRParams.BufferSize, params.SamplingParams.NumSamplingThreads),
			deepcfr.NewReservoirBuffer(dCFRParams.BufferSize, params.SamplingParams.NumSamplingThreads),
		}
		return deepcfr.NewVRSingleDeepCFR(lstm, buffers, baselineBuffers)
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
	gamesRemaining.Add(int64(params.DeepCFRParams.TraversalsPerIter))
	for k := 1; k <= params.DeepCFRParams.TraversalsPerIter; k++ {
		sem <- struct{}{}
		glog.V(2).Infof("[k=%d] Running CFR iteration on random game", k)
		wg.Add(1)
		gamesInProgress.Add(1)
		go func(k int) {
			game := alphacats.NewRandomGame(deck, cardsPerPlayer)
			traversingSampler := sampling.NewMultiOutcomeSampler(
				params.SamplingParams.MaxNumActionsK,
				float32(params.SamplingParams.ExplorationEps))
			notTraversingSampler := sampling.NewOutcomeSampler(
				float32(params.SamplingParams.ExplorationEps))
			walker := cfr.NewVRMCCFR(policy, traversingSampler, notTraversingSampler)
			walker.Run(game)
			glog.V(2).Infof("[k=%d] CFR run complete", k)
			<-sem
			gamesInProgress.Add(-1)
			gamesRemaining.Add(-1)
			wg.Done()
		}(k)
	}

	wg.Wait()
}

type cfrAlgo interface {
	Run(node cfr.GameTreeNode) float32
}

func collectOneSample(walker cfrAlgo, params RunParams) {
	deck, cardsPerPlayer := getDeck(params.DeckType)
	gamesRemaining.Add(int64(params.DeepCFRParams.TraversalsPerIter))
	for k := 1; k <= params.DeepCFRParams.TraversalsPerIter; k++ {
		gamesInProgress.Add(1)
		glog.V(2).Infof("[k=%d] Running CFR iteration on random game", k)
		game := alphacats.NewRandomGame(deck, cardsPerPlayer)
		walker.Run(game)
		glog.V(2).Infof("[k=%d] CFR run complete", k)
		gamesInProgress.Add(-1)
		gamesRemaining.Add(-1)
	}
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
	flag.IntVar(&params.SamplingParams.MaxNumActionsK, "sampling.max_num_actions", 3,
		"Max number of actions to sample for traversing player in multi-outcome sampling")
	flag.Float64Var(&params.SamplingParams.ExplorationEps, "sampling.exploration_eps", 0.1,
		"Exploration factor used in multi-outcome sampling")
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
	flag.IntVar(&params.DeepCFRParams.ModelParams.NumPredictionWorkers,
		"deepcfr.model.num_prediction_workers", 2,
		"Number of worker threads for making predictions on GPU")
	flag.IntVar(&params.DeepCFRParams.ModelParams.MaxTrainingDataWorkers,
		"deepcfr.model.max_training_data_workers", 24,
		"Number of worker threads for training data encoding")
	flag.IntVar(&params.DeepCFRParams.ModelParams.MaxInferenceBatchSize,
		"deepcfr.model.max_inference_batch_size", 3000,
		"Max size of batches for prediction")
	flag.IntVar(&params.RDBParams.BlockCacheCapacity,
		"deepcfr.buffer.block_cache_capacity", 8*MiB,
		"Block cache capacity if using rocksd sampled actions")
	flag.IntVar(&params.RDBParams.WriteBuffer,
		"deepcfr.buffer.write_buffer", 256*MiB,
		"Write buffer if using rocksd sampled actions")
	flag.IntVar(&params.RDBParams.BloomFilterNumBits,
		"deepcfr.buffer.bloom_bits", 10,
		"Number of bits/sample for Bloom filter if using rocksdb reservoir buffer")
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

	traversingSampler := sampling.NewMultiOutcomeSampler(
		params.SamplingParams.MaxNumActionsK,
		float32(params.SamplingParams.ExplorationEps))
	notTraversingSampler := sampling.NewOutcomeSampler(
		float32(params.SamplingParams.ExplorationEps))
	walker := cfr.NewVRMCCFR(policy, traversingSampler, notTraversingSampler)

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
		if params.SamplingParams.NumSamplingThreads > 1 {
			collectSamples(policy, params)
		} else {
			collectOneSample(walker, params)
		}
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
	name := fmt.Sprintf("iter_%08d.policy", iter)
	outputFile := filepath.Join(outputDir, name)
	glog.Infof("[t=%d] Saving current policy to %v", iter, outputFile)
	f, err := os.Create(outputFile)
	if err != nil {
		glog.Fatal(err)
	}
	defer f.Close()

	w := gzip.NewWriter(f)
	defer w.Close()

	buf, err := policy.MarshalBinary()
	if err != nil {
		return err
	}

	_, err = w.Write(buf)
	return err
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

	buf, err := ioutil.ReadAll(r)
	if err != nil {
		glog.Fatal(err)
	}

	policy := newPolicy(params)
	if err := policy.UnmarshalBinary(buf); err != nil {
		glog.Fatal(err)
	}

	glog.Infof("Finished loading policy (took: %v)", time.Since(start))
	return policy
}
