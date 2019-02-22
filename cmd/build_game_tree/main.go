package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr/tree"

	"github.com/timpalpant/alphacats"
)

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	flag.Parse()
	go http.ListenAndServe("localhost:4123", nil)

	rand.Seed(*seed)

	infosets := make(map[string]struct{})
	game := alphacats.NewRandomGame()
	tree.VisitInfoSets(game, func(player int, infoset string) {
		infosets[infoset] = struct{}{}
		if len(infosets)%1000000 == 0 {
			glog.Infof("%d infosets", len(infosets))
		}
	})

	glog.Infof("%d infosets", len(infosets))
}
