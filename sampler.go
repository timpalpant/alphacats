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
	node.buildChildren()
	return node
}

func CountTerminalNodes(root GameNode) int {
	return countTerminalNodesDFS(root)
}

func countTerminalNodesDFS(node GameNode) int {
	if node.IsTerminal() {
		return 1
	}

	node.buildChildren()
	total := 0
	for _, child := range node.children {
		total += countTerminalNodesDFS(child)
	}

	if total%10000 == 0 {
		glog.Infof("Found %d terminal nodes", total)
	}

	// Clear children to allow this subtree to be GC'ed.
	node.children = nil
	return total
}
