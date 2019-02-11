package alphacats

import (
	"math/rand"
	"sort"

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

var totalFound = 0
var totalP0Win = 0
var totalP1Win = 0

func CountTerminalNodes(root GameNode) int {
	return countTerminalNodesDFS(root)
}

func countTerminalNodesDFS(node GameNode) int {
	if node.IsTerminal() {
		totalFound++
		if node.Winner() == gamestate.Player0 {
			totalP0Win++
		} else {
			totalP1Win++
		}

		if totalFound%1000000 == 0 {
			glog.Infof("Found %d terminal nodes (P0 win: %d, P1 win: %d)",
				totalFound, totalP0Win, totalP1Win)
			glog.Infof("Last game: %s", node.GetHistory())
		}

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
