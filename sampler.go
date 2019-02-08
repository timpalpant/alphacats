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
		if i == 0 {
			continue // Deal
		}
		s := fmt.Sprintf("\tTurn %d: %s", i, node)
		result = append(result, s)
	}

	return strings.Join(result, "\n")
}

func CountInfoSets(root *GameNode) int {
	seen := make(map[GameState]struct{})
	allInfoSets := make(map[InfoSet]struct{})
	walkGameTree(root, seen, allInfoSets)
	return len(allInfoSets)
}

func walkGameTree(node *GameNode, seen map[GameState]struct{}, infoSets map[InfoSet]struct{}) {
	if _, ok := seen[node.state]; ok {
		return // Already processed this subtree from another history.
	}

	infoSets[node.state.Player0Info] = struct{}{}
	if len(infoSets)%100000 == 0 {
		glog.Infof("Found %d distinct info sets", len(infoSets))
	}

	defer func() {
		if r := recover(); r != nil {
			glog.Errorf("%s", node)
			panic(r)
		}
	}()

	node.buildChildren()
	for _, child := range node.children {
		walkGameTree(child, seen, infoSets)
	}

	// We've fully walked this sub-tree, so clear the children to free memory.
	node.children = nil
	seen[node.state] = struct{}{}
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
