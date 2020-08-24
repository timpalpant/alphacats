package model

import (
	"bytes"
	"encoding/gob"

	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/mcts"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/gamestate"
)

// SmoothUCTPolicy uses SmoothUCT tree search to select positions (Heinreich and Silver, 2015).
// Before selecting a policy, additional tree searches are performed for any nodes in the history
// that have less than the given number of searches, starting from the root.
type SmoothUCTPolicy struct {
	search              *mcts.SmoothUCT
	temperature         float32
	numSearchIterations int
}

func NewSmoothUCTPolicy(search *mcts.SmoothUCT, temperature float32, numSearchIterations int) *SmoothUCTPolicy {
	return &SmoothUCTPolicy{
		search:              search,
		temperature:         temperature,
		numSearchIterations: numSearchIterations,
	}
}

func (p *SmoothUCTPolicy) GetPolicy(node cfr.GameTreeNode) []float32 {
	opponentPolicy := func(node cfr.GameTreeNode) []float32 {
		return p.search.GetPolicy(node, p.temperature)
	}
	ancestors := getAncestors(node)
	rootNode := ancestors[len(ancestors)-1]
	player := gamestate.Player(node.Player())
	infoSet := rootNode.(*alphacats.GameNode).GetInfoSet(player)
	beliefs := alphacats.NewBeliefState(opponentPolicy, infoSet)
	// Walk down the tree from the root, performing simulations at each node.
	for i := len(ancestors) - 1; i >= 0; i-- {
		currentNode := ancestors[i]
		infoSet = currentNode.(*alphacats.GameNode).GetInfoSet(player)
		beliefs.Update(infoSet)

		nSearches := p.numSearchIterations - p.search.GetVisitCount(currentNode)
		for j := 0; j < nSearches; j++ {
			game := beliefs.SampleDeterminization()
			p.search.Run(game)
		}
	}

	return p.search.GetPolicy(node, p.temperature)
}

func getAncestors(node cfr.GameTreeNode) []cfr.GameTreeNode {
	var result []cfr.GameTreeNode
	for node.Parent() != nil {
		result = append(result, node)
		node = node.Parent()
	}

	result = append(result, node)
	return result
}

func (p *SmoothUCTPolicy) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(p.search); err != nil {
		return nil, err
	}
	if err := enc.Encode(p.temperature); err != nil {
		return nil, err
	}
	if err := enc.Encode(p.numSearchIterations); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (p *SmoothUCTPolicy) GobDecode(buf []byte) error {
	r := bytes.NewReader(buf)
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&p.search); err != nil {
		return err
	}
	if err := dec.Decode(&p.temperature); err != nil {
		return err
	}
	if err := dec.Decode(&p.numSearchIterations); err != nil {
		return err
	}
	return nil
}
