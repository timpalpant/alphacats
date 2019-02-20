package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
)

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	flag.Parse()

	rand.Seed(*seed)
	go http.ListenAndServe("localhost:4123", nil)

	opt := cfr.New(cfr.Params{
		SampleChanceNodes: true,
	})
	game := alphacats.NewRandomGame()
	expectedValue := opt.Run(game)
	glog.Infof("Expected value is: %v", expectedValue)
}
