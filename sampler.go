package alphacats

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"

	"github.com/golang/glog"
)

type Strategy interface {
	Select(nChoices int) int
}

type History []*GameNode

func (h History) String() string {
	var result []string
	result = append(result, fmt.Sprintf("Game of %d turns:", len(h)))
	for i, node := range h {
		s := fmt.Sprintf("\tTurn %d: %v", i, node)
		result = append(result, s)
	}

	return strings.Join(result, "\n")
}

func SampleHistory(root *GameNode, s Strategy) History {
	var history History
	node := root
	for node != nil {
		glog.Infof("%+v", node)
		history = append(history, node)
		glog.Infof("Sampling one of %d children", node.NumChildren())
		node = SampleOne(node, s)
	}

	return history
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
