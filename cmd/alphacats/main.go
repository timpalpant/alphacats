package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/model"
)

func main() {
	params := model.Params{}
	seed := flag.Int64("seed", 123, "Random seed")
	iter := flag.Int("iter", 100, "Number of DeepCFR iterations to perform")
	bufSize := flag.Int("buf_size", 10000000, "Size of reservoir sample buffer")
	traversalsPerIter := flag.Int("traversals_per_iter", 10000000,
		"Number of OS-CFR traversals to perform each iteration")
	explorationDelta := flag.Float64("exploration_delta", 0.1,
		"Fraction of time to explore randomly off-policy")
	flag.IntVar(&params.BatchSize, "batch_size", 4096,
		"Size of minibatches to save for training")
	flag.StringVar(&params.ModelOutputDir, "model_dir", "",
		"Directory to save trained models to")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	lstm := model.NewLSTM(params)
	buffer := deepcfr.NewReservoirBuffer(*bufSize)
	deepCFR := deepcfr.New(lstm, buffer)
	opt := cfr.NewOutcomeSampling(deepCFR, float32(*explorationDelta))

	for t := 1; t <= *iter; t++ {
		glog.Infof("[t=%d] Collecting samples", t)
		for k := 1; k <= *traversalsPerIter; k++ {
			glog.V(3).Infof("[k=%d] Running OS-CFR on random game", k)
			game := alphacats.NewRandomGame()
			opt.Run(game)
		}

		glog.Infof("[t=%d] Training network", t)
		deepCFR.Update()
	}
}
