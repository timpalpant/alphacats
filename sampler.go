package alphacats

import (
	"math/rand"
	"runtime"
	"sort"
	"sync"

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
	for _, child := range root.children {
		wg.Add(1)
		sem <- struct{}{}
		go func(child GameNode) {
			myCount := countTerminalNodesDFS(child)

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

func countTerminalNodesDFS(node GameNode) int {
	if node.IsTerminal() {
		return 1
	}

	node.BuildChildren()
	total := 0
	for _, child := range node.children {
		total += countTerminalNodesDFS(child)
	}

	// Clear children to allow this subtree to be GC'ed.
	node.Clear()
	return total
}
