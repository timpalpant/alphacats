package alphacats

import (
	"expvar"
	"fmt"
	"math/rand"

	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

var (
	nodesVisited         = expvar.NewInt("nodes_visited")
	terminalNodesVisited = expvar.NewInt("nodes_visited/terminal")
	playerNodesVisited   = expvar.NewInt("nodes_visited/player")
	chanceNodesVisited   = expvar.NewInt("nodes_visited/chance")
)

// turnType represents the kind of turn at a given point in the game.
type turnType uint8

const (
	_ turnType = iota
	PlayTurn
	GiveCard
	ShuffleDrawPile
	MustDefuse
	InsertKittenRandom
	GameOver
)

var turnTypeStr = [...]string{
	"Invalid",
	"PlayTurn",
	"GiveCard",
	"ShuffleDrawPile",
	"MustDefuse",
	"InsertKittenRandom",
	"GameOver",
}

func (tt turnType) String() string {
	return turnTypeStr[tt]
}

// GameNode implements cfr.GameTreeNode for Exploding Kittens.
// GameNode represents a state of play in the extensive-form game tree.
type GameNode struct {
	state    gamestate.GameState
	player   gamestate.Player
	turnType turnType
	// pendingTurns is the number of turns the player has outstanding
	// to play. In general this will be 1, except when Slap cards are played.
	pendingTurns int
	// nDrawPileCards is used lazily be ShuffleDrawPile nodes to cache
	// the number of cards in the draw pile.
	nDrawPileCards int

	// children are the possible next states in the game.
	// Which child is realized will depend on chance or a player's action.
	children []GameNode
	// actions are the action taken by the player to reach each child.
	// len(actions) must always equal len(children).
	actions []gamestate.Action
	parent  *GameNode

	rng    *rand.Rand
	gnPool *gameNodeSlicePool
	aPool  *actionSlicePool
}

// Verify that we implement the interface.
var _ cfr.GameTreeNode = &GameNode{}

// NewGame creates a root node for a new game with the given draw pile
// and hands dealt to each player.
func NewGame(drawPile cards.Stack, p0Deal, p1Deal cards.Set) *GameNode {
	return &GameNode{
		state: gamestate.New(drawPile, p0Deal, p1Deal),
		// Player0 always goes first.
		player:       gamestate.Player0,
		turnType:     PlayTurn,
		pendingTurns: 1,
		rng:          rand.New(rand.NewSource(rand.Int63())),
		gnPool:       &gameNodeSlicePool{},
		aPool:        &actionSlicePool{},
	}
}

// NewRandomGame creates a root node for a new random game, as if the
// deck were shuffled and each player were dealt a random hand of cards.
func NewRandomGame(deck []cards.Card, cardsPerPlayer int) *GameNode {
	rand.Shuffle(len(deck), func(i, j int) {
		deck[i], deck[j] = deck[j], deck[i]
	})

	p0Deal := cards.NewSetFromCards(deck[:cardsPerPlayer])
	p0Deal.Add(cards.Defuse)
	p1Deal := cards.NewSetFromCards(deck[cardsPerPlayer : 2*cardsPerPlayer])
	p1Deal.Add(cards.Defuse)
	drawPile := cards.NewStackFromCards(deck[2*cardsPerPlayer:])
	randPos := rand.Intn(drawPile.Len() + 1)
	drawPile.InsertCard(cards.ExplodingCat, randPos)
	randPos = rand.Intn(drawPile.Len() + 1)
	drawPile.InsertCard(cards.Defuse, randPos)
	return NewGame(drawPile, p0Deal, p1Deal)
}

// Type implements cfr.GameTreeNode.
func (gn *GameNode) Type() cfr.NodeType {
	switch gn.turnType {
	case ShuffleDrawPile, InsertKittenRandom:
		return cfr.ChanceNodeType
	case GameOver:
		return cfr.TerminalNodeType
	default:
		return cfr.PlayerNodeType
	}
}

// Player implements cfr.GameTreeNode.
func (gn *GameNode) Player() int {
	return int(gn.player)
}

func (gn *GameNode) GetHistory() gamestate.History {
	return gn.state.GetHistory()
}

func (gn *GameNode) LastAction() gamestate.Action {
	return gn.state.LastAction()
}

// InfoSet implements cfr.GameTreeNode.
func (gn *GameNode) InfoSet(player int) cfr.InfoSet {
	if gn.children == nil {
		gn.buildChildren()
	}

	return &InfoSetWithAvailableActions{
		InfoSet:          gn.state.GetInfoSet(gamestate.Player(player)),
		AvailableActions: gn.actions,
	}
}

// Utility implements cfr.GameTreeNode.
func (gn *GameNode) Utility(player int) float64 {
	if gn.Type() != cfr.TerminalNodeType {
		panic("cannot get the utility of a non-terminal node")
	}

	if int(gn.player) == player {
		return 1.0
	}

	return -1.0
}

// String implements fmt.Stringer.
func (gn *GameNode) String() string {
	return fmt.Sprintf("%v's turn to %v (%d remaining). Hand: %s. %d cards in draw pile: %s",
		gn.player, gn.turnType, gn.pendingTurns, gn.state.GetPlayerHand(gn.player),
		gn.state.GetDrawPile().Len(), gn.state.GetDrawPile())
}

func (gn *GameNode) GetDrawPile() cards.Stack {
	return gn.state.GetDrawPile()
}

func (gn *GameNode) allocChildren(n int) {
	gn.children = gn.gnPool.alloc(n)
	gn.actions = gn.aPool.alloc(n)
	// Children are initialized as a copy of the current game node,
	// but without any children (the new node's children must be built).
	childPrototype := *gn
	childPrototype.children = nil
	childPrototype.actions = nil
	childPrototype.parent = gn
	for i := 0; i < n; i++ {
		gn.children = append(gn.children, childPrototype)
		gn.actions = append(gn.actions, gamestate.Action{})
	}
}

func (gn *GameNode) buildChildren() {
	if len(gn.children) > 0 {
		return // Already built.
	}

	switch gn.turnType {
	case PlayTurn:
		gn.buildPlayTurnChildren()
	case GiveCard:
		gn.buildGiveCardChildren()
	case ShuffleDrawPile:
		// Shuffle children are lazily generated since the
		// number of children may be large and in chance sampling
		// CFR we are only going to choose one of them.
		gn.allocChildren(1)
	case InsertKittenRandom:
		gn.buildInsertKittenRandomChildren()
	case MustDefuse:
		gn.buildMustDefuseChildren()
	case GameOver:
	default:
		panic("unimplemented turn type!")
	}
}

func (gn *GameNode) NumChildren() int {
	// Chance children are lazily generated because we always sample them
	// but we can easily compute how many there will be.
	if gn.turnType == ShuffleDrawPile {
		return factorial[gn.nDrawPileCards]
	}

	if gn.children == nil {
		gn.buildChildren()
	}

	if len(gn.children) != len(gn.actions) {
		panic(fmt.Errorf("%d children, %d actions: %v",
			len(gn.children), len(gn.actions), gn))
	}

	return len(gn.children)
}

// GetChild implements cfr.GameTreeNode.
func (gn *GameNode) GetChild(i int) cfr.GameTreeNode {
	if gn.children == nil {
		gn.buildChildren()
	}

	if gn.turnType == ShuffleDrawPile {
		shuffle := nthShuffle(gn.state.GetDrawPile(), i)
		return gn.buildShuffleChild(shuffle)
	}

	return &gn.children[i]
}

func (gn *GameNode) Parent() cfr.GameTreeNode {
	return gn.parent
}

// GetChildProbability implements cfr.GameTreeNode.
func (gn *GameNode) GetChildProbability(i int) float64 {
	if gn.Type() != cfr.ChanceNodeType {
		panic("cannot get the probability of a non-chance node")
	}

	// Uniform random over all possible shuffles / insertion spots.
	return 1.0 / float64(gn.NumChildren())
}

// SampleChild implements cfr.GameTreeNode.
func (gn *GameNode) SampleChild() (cfr.GameTreeNode, float64) {
	// All chance nodes are uniform random over their children.
	selected := gn.rng.Intn(gn.NumChildren())
	return gn.GetChild(selected), gn.GetChildProbability(selected)
}

// Close implements cfr.GameTreeNode.
func (gn *GameNode) Close() {
	nodesVisited.Add(1)
	switch gn.Type() {
	case cfr.TerminalNodeType:
		terminalNodesVisited.Add(1)
	case cfr.PlayerNodeType:
		playerNodesVisited.Add(1)
	case cfr.ChanceNodeType:
		chanceNodesVisited.Add(1)
	}

	gn.gnPool.free(gn.children)
	gn.children = nil
	gn.aPool.free(gn.actions)
	gn.actions = nil
}

func makePlayTurnNode(node *GameNode, player gamestate.Player, pendingTurns int) {
	if node.state.GetPlayerHand(player).Contains(cards.ExplodingCat) {
		// Player drew an exploding kitten, must defuse it before continuing.
		if node.state.GetPlayerHand(player).Contains(cards.Defuse) {
			// Player has a defuse card, must play it.
			makeMustDefuseNode(node, player, pendingTurns)
		} else {
			// Player does not have a defuse card, end game with loss for them.
			winner := nextPlayer(player)
			makeTerminalGameNode(node, winner)
		}
	} else {
		// Just a normal card, add it to player's hand and continue.
		if pendingTurns <= 0 {
			// Player's turn is done, next player.
			player = nextPlayer(player)
			pendingTurns = 1
		}

		node.player = player
		node.turnType = PlayTurn
		node.pendingTurns = pendingTurns
	}
}

func makeGiveCardNode(node *GameNode, player gamestate.Player) {
	node.player = player
	node.turnType = GiveCard
}

func makeMustDefuseNode(node *GameNode, player gamestate.Player, pendingTurns int) {
	node.player = player
	node.turnType = MustDefuse
	node.pendingTurns = pendingTurns
	node.state.Apply(gamestate.Action{
		Player: player,
		Type:   gamestate.PlayCard,
		Card:   cards.Defuse,
	})
}

func makeTerminalGameNode(node *GameNode, winner gamestate.Player) {
	node.player = winner
	node.turnType = GameOver
}

func (gn *GameNode) buildPlayTurnChildren() {
	hand := gn.state.GetPlayerHand(gn.player)
	gn.allocChildren(hand.Len() + 1)
	i := 0
	// Play one of the cards in our hand.
	hand.Iter(func(card cards.Card, count uint8) {
		child := &gn.children[i]
		action := gamestate.Action{
			Player: gn.player,
			Type:   gamestate.PlayCard,
			Card:   card,
		}
		gn.actions[i] = action
		child.state.Apply(action)

		switch card {
		case cards.Defuse, cards.SeeTheFuture:
			makePlayTurnNode(child, gn.player, gn.pendingTurns)
		case cards.Skip, cards.DrawFromTheBottom:
			// Ends our current turn (with/without drawing a card).
			makePlayTurnNode(child, gn.player, gn.pendingTurns-1)
		case cards.Shuffle:
			child.turnType = ShuffleDrawPile
			child.nDrawPileCards = gn.state.GetDrawPile().Len()
		case cards.Slap1x, cards.Slap2x:
			// Ends our turn (and all pending turns). Goes to next player with
			// any pending turns + slap.
			pendingTurns := 1
			if card == cards.Slap2x {
				pendingTurns = 2
			}

			lastAction := gn.state.LastAction()
			slapBack := lastAction.Type == gamestate.PlayCard && (lastAction.Card == cards.Slap1x || lastAction.Card == cards.Slap2x)
			if slapBack {
				pendingTurns += gn.pendingTurns
			}

			makePlayTurnNode(child, nextPlayer(gn.player), pendingTurns)
		case cards.Cat:
			if child.state.GetPlayerHand(nextPlayer(gn.player)).Len() == 0 {
				// Other player has no cards in their hand, this was a no-op.
				makePlayTurnNode(child, gn.player, gn.pendingTurns)
			} else {
				// Other player must give us a card.
				makeGiveCardNode(child, nextPlayer(gn.player))
			}
		default:
			panic(fmt.Errorf("Player playing unsupported %v card", card))
		}

		i++
	})

	gn.children = gn.children[:i+1]
	gn.actions = gn.actions[:i+1]
	// End our turn by drawing a card.
	lastChild := &gn.children[i]
	action := gamestate.Action{
		Player: gn.player,
		Type:   gamestate.DrawCard,
	}
	lastChild.state.Apply(action)
	gn.actions[i] = action
	makePlayTurnNode(lastChild, gn.player, gn.pendingTurns-1)
}

func (gn *GameNode) buildGiveCardChildren() {
	hand := gn.state.GetPlayerHand(gn.player)
	gn.allocChildren(hand.Len())
	i := 0
	hand.Iter(func(card cards.Card, count uint8) {
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Adding card to opponent's hand,
		//   3) Returning to opponent's turn.
		child := &gn.children[i]
		action := gamestate.Action{
			Player: gn.player,
			Type:   gamestate.GiveCard,
			Card:   card,
		}
		child.state.Apply(action)
		gn.actions[i] = action

		// Game play returns to other player (with the given card in their hand).
		makePlayTurnNode(child, nextPlayer(gn.player), gn.pendingTurns)

		i++
	})

	gn.children = gn.children[:i]
	gn.actions = gn.actions[:i]
}

func (gn *GameNode) buildMustDefuseChildren() {
	// 1 card in draw pile -> nOptions = 2 -> 3 children -> i in 0, 1 + prune extra child
	// 5 card in draw pile -> nOptions = 6 -> 7 children -> i in 0..5 + prune extra child
	// 6 card in draw pile -> nOptions = 6 -> 7 children -> i in 0..5 + use extra child for bottom
	nCardsInDrawPile := gn.state.GetDrawPile().Len()
	nOptions := min(nCardsInDrawPile+1, 6)
	gn.allocChildren(nOptions + 2)
	// Place in the i'th position.
	for i := 0; i < nOptions; i++ {
		child := &gn.children[i]
		action := gamestate.Action{
			Player:             gn.player,
			Type:               gamestate.InsertExplodingCat,
			PositionInDrawPile: uint8(i + 1),
		}
		child.state.Apply(action)
		gn.actions[i] = action

		makePlayTurnNode(child, gn.player, gn.pendingTurns)
	}

	// Place randomly.
	child := &gn.children[nOptions]
	child.turnType = InsertKittenRandom
	gn.actions[nOptions] = gamestate.Action{
		Player: gn.player,
		Type:   gamestate.InsertExplodingCat,
	}

	// Place exploding cat on the bottom of the draw pile.
	if nCardsInDrawPile > 5 {
		child := &gn.children[len(gn.children)-1]
		action := gamestate.Action{
			Player:             gn.player,
			Type:               gamestate.InsertExplodingCat,
			PositionInDrawPile: uint8(nCardsInDrawPile + 1), // bottom
		}
		child.state.Apply(action)
		gn.actions[len(gn.children)-1] = action
		makePlayTurnNode(child, gn.player, gn.pendingTurns)
	} else {
		gn.children = gn.children[:len(gn.children)-1]
		gn.actions = gn.actions[:len(gn.actions)-1]
	}
}

func (gn *GameNode) buildInsertKittenRandomChildren() {
	nPositions := gn.state.GetDrawPile().Len() + 1
	gn.allocChildren(nPositions)
	for i := 0; i < nPositions; i++ {
		child := &gn.children[i]
		action := gamestate.Action{
			Player:             gn.player,
			Type:               gamestate.InsertExplodingCat,
			PositionInDrawPile: uint8(i + 1),
		}
		child.state.Apply(action)
		gn.actions[i] = action

		makePlayTurnNode(child, gn.player, gn.pendingTurns)
	}
}

func (gn *GameNode) buildShuffleChild(newDrawPile cards.Stack) *GameNode {
	result := &gn.children[0]
	result.state = gamestate.NewShuffled(result.state, newDrawPile)
	result.children = nil
	result.turnType = PlayTurn
	return result
}

func nextPlayer(p gamestate.Player) gamestate.Player {
	if p != gamestate.Player0 && p != gamestate.Player1 {
		panic(fmt.Sprintf("cannot call nextPlayer with player %v", p))
	}

	return 1 - p
}

func min(i, j int) int {
	if i < j {
		return i
	}

	return j
}
