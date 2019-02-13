package alphacats

import (
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/glog"

	"github.com/timpalpant/alphacats/internal/gamestate"
)

type Strategy interface {
	Select(nChoices int) int
}

func SampleHistory(root GameNode, s Strategy) []gamestate.Action {
	node := root
	for !node.IsTerminal() {
		node = SampleOne(node, s)
	}

	return node.GetHistory()
}

func SampleOne(gn GameNode, s Strategy) GameNode {
	var selected int
	if gn.turnType.IsChance() {
		x := rand.Float64()
		selected = sort.Search(len(gn.children), func(i int) bool {
			return gn.cumulativeProbs[i] >= x
		})
	} else {
		selected = s.Select(len(gn.children))
	}

	node := gn.children[selected]
	node.BuildChildren()
	return node
}

func CountTerminalNodes(root GameNode) int {
	// Parallelize on the first node's children (initial deals).
	nProcessed := 0
	result := 0
	mu := sync.Mutex{}
	wg := sync.WaitGroup{}
	// Limit parallelism to number of CPUs.
	sem := make(chan struct{}, runtime.NumCPU())
	startTime = time.Now()
	for _, child := range root.children {
		wg.Add(1)
		sem <- struct{}{}
		go func(child GameNode) {
			child.gnPool = &gameNodeSlicePool{}
			child.fPool = &floatSlicePool{}
			myCount := countTerminalNodesDFS(&child)

			mu.Lock()
			nProcessed++
			result += myCount
			glog.Infof("Processed %d out of %d children (%d terminal nodes)",
				nProcessed, root.NumChildren(), result)
			mu.Unlock()

			<-sem
			wg.Done()
		}(child)
	}

	wg.Wait()
	return result
}

var startTime time.Time
var totalTerminalNodes int64

func countTerminalNodesDFS(node *GameNode) int {
	if node.IsTerminal() {
		if n := atomic.AddInt64(&totalTerminalNodes, 1); n%1000000 == 0 {
			gps := float64(n) / time.Since(startTime).Seconds()
			glog.Infof("%d terminal nodes (%.1f games/s)", n, gps)
		}

		return 1
	}

	node.BuildChildren()
	total := 0
	for i := range node.children {
		total += countTerminalNodesDFS(&node.children[i])
	}

	// Clear children to allow this subtree to be GC'ed.
	node.Reset()
	return total
}
