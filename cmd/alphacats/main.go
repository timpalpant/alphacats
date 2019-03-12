package main

import (
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"

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
	case "none":
		return cfr.New(policy)
	default:
		panic(fmt.Errorf("unknown sampling type: %v", samplingType))
	}

	return nil
}

func getPolicy(cfrType string, params model.Params, bufSize int) cfr.StrategyProfile {
	switch cfrType {
	case "tabular":
		return cfr.NewStrategyTable(cfr.DiscountParams{})
	case "deep":
		lstm := model.NewLSTM(params)
		buffer := deepcfr.NewReservoirBuffer(bufSize)
		return deepcfr.New(lstm, buffer)
	default:
		panic(fmt.Errorf("unknown CFR type: %v", cfrType))
	}
}

func main() {
	params := model.Params{}
	cfrType := flag.String("cfrtype", "tabular", "Type of CFR to run (tabular, deep)")
	samplingType := flag.String("sampling", "external",
		"Type of sampling to perform (external, chance, outcome, none)")
	seed := flag.Int64("seed", 123, "Random seed")
	iter := flag.Int("iter", 100, "Number of DeepCFR iterations to perform")
	bufSize := flag.Int("buf_size", 10000000, "Size of reservoir sample buffer")
	traversalsPerIter := flag.Int("traversals_per_iter", 10000000,
		"Number of OS-CFR traversals to perform each iteration")
	outputDir := flag.String("output_dir", "", "Directory to save policies to")
	flag.IntVar(&params.BatchSize, "batch_size", 4096,
		"Size of minibatches to save for training")
	flag.StringVar(&params.ModelOutputDir, "model_dir", "",
		"Directory to save trained models to")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	if err := os.MkdirAll(*outputDir, 0777); err != nil {
		glog.Fatal(err)
	}

	deck := []cards.Card{
		cards.Shuffle, cards.SeeTheFuture, cards.Slap1x, cards.Slap2x,
		cards.Skip, cards.Cat, cards.Skip, cards.DrawFromTheBottom,
		cards.Slap1x, cards.Cat, cards.SeeTheFuture,
	}

	policy := getPolicy(*cfrType, params, *bufSize)
	opt := getCFRAlgo(policy, *samplingType)

	for t := 1; t <= *iter; t++ {
		glog.Infof("[t=%d] Collecting samples", t)
		for k := 1; k <= *traversalsPerIter; k++ {
			glog.V(3).Infof("[k=%d] Running ES-CFR on random game", k)
			game := alphacats.NewRandomGame(deck)
			opt.Run(game)
		}

		glog.Infof("[t=%d] Training network", t)
		policy.Update()

		// TODO(palpant): Implement marshalling for DeepCFR policy,
		// which should save the current sample buffers + trained models.
		if policy, ok := policy.(*cfr.StrategyTable); ok {
			if err := savePolicy(policy, *outputDir, t); err != nil {
				glog.Fatal(err)
			}
		}
	}
}

func savePolicy(policy *cfr.StrategyTable, outputDir string, iter int) error {
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

	return policy.MarshalTo(w)
}
