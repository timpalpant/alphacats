package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"sync"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/tree"

	"github.com/timpalpant/alphacats"
)

func main() {
	seed := flag.Int64("seed", 123, "Random seed")
	flag.Parse()
	go http.ListenAndServe("localhost:4123", nil)

	rand.Seed(*seed)

	game := alphacats.NewRandomGame()
	game.BuildChildren() // Build first player choices.
	var wg sync.WaitGroup
	var mu sync.Mutex
	nInfoSets := game.NumChildren()
	// Parallelize over first two levels of children, excluding chance nodes.
	for i := 0; i < game.NumChildren(); i++ {
		child := game.GetChild(i)
		child.BuildChildren()
		for j := 0; j < child.NumChildren(); j++ {
			grandchild := child.GetChild(j)
			if child.Type() == cfr.ChanceNode && j > 0 {
				break
			}

			grandchild.(*alphacats.GameNode).Liberate()
			wg.Add(1)
			go func(i, j int) {
				glog.Infof("[child=%d,%d] Starting infoset iteration", i, j)
				myInfoSets := make(map[string]struct{})
				tree.VisitInfoSets(grandchild, func(player int, infoset string) {
					myInfoSets[infoset] = struct{}{}
					if len(myInfoSets)%1000000 == 0 {
						glog.Infof("[child=%d,%d] %d infosets", i, j, len(myInfoSets))
					}
				})

				glog.Infof("[child=%d,%d] Infoset iteration completed: %d infosets",
					i, j, len(myInfoSets))
				mu.Lock()
				defer mu.Unlock()
				nInfoSets += len(myInfoSets) + 1
			}(i, j)
		}
	}

	wg.Wait()
	glog.Infof("%d infosets total", nInfoSets)
}
