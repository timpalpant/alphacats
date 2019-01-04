package alphacats

type GameState struct {
	drawPile     CardPile
	ourHand      CardSet
	opponentHand CardSet
}

func EnumerateInitialGameStates() []GameState {

}

type GameTree struct {
	rootNodes []Node
}

type Node interface {
}

type ChanceNode struct {
	nextNodes               []Node
	cumulativeProbabilities []float64
}

type ActionNode struct {
	infoSet   InfoSet
	nextNodes map[Action]Node
}

type TerminalNode float64

const (
	Win  TerminalNode = 1.0
	Lose TerminalNode = -1.0
)

func BuildGameTree() *GameTree {
	initialInfoSets := EnumerateInitialInfoSets()
	rootNodes := make([]Node, 0, len(initialInfoSets))
	for _, infoSet := range initialInfoSets {
		actions := availableTurnActions(infoSet)

		node := &ActionNode{
			infoSet:   infoSet,
			nextNodes: buildNextNodes(infoSet, actions),
		}

		rootNodes = append(rootNodes, node)
	}

	return &GameTree{
		rootNodes: rootNodes,
	}
}

func buildNextNodes(infoSet InfoSet, actions []Action) map[Action]Node {
	result := make(map[Action]Node, len(actions))
	for _, action := range actions {
		node := buildNextNode(infoSet, action)
		result[action] = node
	}

	return result
}

func buildNextNode(infoSet InfoSet, action Action) Node {
	switch action {
	case DrawCard:
		return buildChanceNode(InfoSet)
	case PlayDefuseCard:
		infoSet.OurHand[Defuse]--
	case PlaySkipCard:
	case PlaySlap1xCard:
	case PlaySlap2xCard:
	case PlaySeeTheFutureCard:
	case PlayShuffleCard:
	case PlayDrawFromTheBottomCard:
	case PlayCatCard:
	}
}

func buildChanceNode(infoSet InfoSet) {
	return &ChanceNode{
		nextNodes:               children,
		cumulativeProbabilities: probs,
	}
}

func availableTurnActions(infoSet InfoSet) []Action {
	result := make([]Action, 0)

	// Can choose to draw a card and end our turn.
	result = append(result, DrawCard)

	// Can choose to play any of the cards in our hand.
	for _, card := range infoSet.OurHand.AsSlice() {
		action := GetPlayActionForCard(card)
		result = append(result, action)
	}

	return result
}
