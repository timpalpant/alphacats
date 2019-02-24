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
	seed := flag.Int64("seed", 123, "Random seed")
	iter := flag.Int("iter", 100, "Number of DeepCFR iterations to perform")
	bufSize := flag.Int("buf_size", 10000000, "Size of reservoir sample buffer")
	traversalsPerIter := flag.Int("traversals_per_iter", 10,
		"Number of ES-CFR traversals to perform each iteration")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	lstm := model.NewLSTM(model.DefaultParams())
	buffer := deepcfr.NewReservoirBuffer(*bufSize)
	deepCFR := deepcfr.New(lstm, buffer)
	opt := cfr.New(cfr.SamplingParams{
		SampleChanceNodes:     true,
		SampleOpponentActions: true,
	}, deepCFR)

	for t := 1; t <= *iter; t++ {
		glog.Infof("[t=%d] Collecting samples", t)
		for k := 1; k <= *traversalsPerIter; k++ {
			glog.Infof("[k=%d] Running ES-CFR on random game", k)
			game := alphacats.NewRandomGame()
			opt.Run(game)
		}

		glog.Infof("[t=%d]  Training network", t)
		deepCFR.Update()
	}
}
