package alphacats

import (
	"math/rand"
	"sort"

	"github.com/timpalpant/alphacats/gamestate"
)

type Strategy interface {
	Select(nChoices int) int
}

func SampleHistory(root *GameNode, s Strategy) []gamestate.Action {
	node := root
	for node != nil {
		node = SampleOne(node, s)
	}

	return node.GetHistory()
}

func SampleOne(gn *GameNode, s Strategy) *GameNode {
	if gn.NumChildren() == 0 {
		return nil
	}

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
