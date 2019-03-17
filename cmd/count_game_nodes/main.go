// Generate and save samples of exploding kitten position for analysis in Python.
package main

import (
	"flag"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"sync"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

func main() {
	flag.Parse()

	go http.ListenAndServe("localhost:4124", nil)

	deck := cards.CoreDeck.AsSlice()
	game := alphacats.NewRandomGame(deck, 4)
	total := countNodes(game, 0)
	glog.Infof("%d nodes in game", total)
}

func countNodes(node cfr.GameTreeNode, depth int) int {
	switch node.Type() {
	case cfr.ChanceNode:
		child, _ := node.SampleChild()
		total := countNodes(child, depth+1) + 1
		node.Close()
		return total
	case cfr.PlayerNode:
		if node.Player() == 0 {
			if depth < 5 {
				var wg sync.WaitGroup
				var mu sync.Mutex
				total := 1
				for i := 0; i < node.NumChildren(); i++ {
					child := node.GetChild(i)
					child.(*alphacats.GameNode).Liberate()
					wg.Add(1)
					go func() {
						result := countNodes(child, depth+1)
						mu.Lock()
						total += result
						mu.Unlock()
						wg.Done()
					}()
				}

				wg.Wait()
				node.Close()
				return total
			} else {
				total := 1
				for i := 0; i < node.NumChildren(); i++ {
					child := node.GetChild(i)
					total += countNodes(child, depth+1)
				}

				node.Close()
				return total
			}
		} else {
			selected := rand.Intn(node.NumChildren())
			child := node.GetChild(selected)
			total := countNodes(child, depth+1) + 1
			node.Close()
			return total
		}
	}

	return 1
}
