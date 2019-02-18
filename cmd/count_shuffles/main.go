package main

import (
	"flag"
	"net/http"
	_ "net/http/pprof"
	"time"

	"github.com/golang/glog"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	flag.Parse()
	go http.ListenAndServe("localhost:4123", nil)

	count := 0
	start := time.Now()
	alphacats.EnumerateGames(func(drawPile cards.Stack, p0Deal, p1Deal cards.Set) {
		count++
		if count%10000000 == 0 {
			gps := float64(count) / time.Since(start).Seconds()
			glog.Infof("%d games (%.1f games/sec)", count, gps)
		}
	})
}
