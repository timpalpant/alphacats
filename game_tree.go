package alphacats

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

// turnType represents the kind of turn at a given point in the game.
type turnType uint8

const (
	_ turnType = iota
	DrawCard
	DrawCardFromBottom
	Deal
	PlayTurn
	GiveCard
	MustDefuse
	SeeTheFuture
	GameOver
)

var turnTypeStr = [...]string{
	"Invalid",
	"DrawCard",
	"DrawCardFromBottom",
	"Deal",
	"PlayTurn",
	"GiveCard",
	"MustDefuse",
	"SeeTheFuture",
	"GameOver",
}

func (tt turnType) IsChance() bool {
	return tt == DrawCard || tt == DrawCardFromBottom || tt == Deal || tt == SeeTheFuture
}

func (tt turnType) String() string {
	return turnTypeStr[tt]
}

// ExplodingKittens implements cfr.ExtensiveFormGame for 2-player ExplodingKittens.
type ExplodingKittens struct{}

// Verify that we implement the inteface.
var _ cfr.ExtensiveFormGame = ExplodingKittens{}

func NewGame() ExplodingKittens {
	return ExplodingKittens{}
}

func (ek ExplodingKittens) NumPlayers() int {
	return 2
}

func (ek ExplodingKittens) RootNode() cfr.GameTreeNode {
	return newRootNode()
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

	// children are the possible next states in the game.
	// Which child is realized will depend on chance or a player's action.
	children []GameNode
	// If turnType is Chance, then these are the probabilities of selecting
	// each of the children, i.e. children[i] occurs with probabilities[i].
	// For non-chance nodes probabilities is nil.
	probabilities []float64

	gnPool *gameNodeSlicePool
	fPool  *floatSlicePool
}

// Verify that we implement the inteface.
var _ cfr.GameTreeNode = &GameNode{}

func newRootNode() *GameNode {
	return &GameNode{
		player:   gamestate.Player0,
		turnType: Deal,
	}
}

// Reset clears the space within this GameNode allocated for children.
// It is not necessary to call Reset, but may be done to free the memory
// used by the sub-tree rooted at this node. Children will be automatically
// rebuilt as needed.
func (gn *GameNode) Reset() {
	gn.gnPool.free(gn.children)
	gn.children = nil
	gn.fPool.free(gn.probabilities)
	gn.probabilities = nil
}

// IsChance implements cfr.GameTreeNode.
func (gn *GameNode) IsChance() bool {
	return gn.turnType.IsChance()
}

func (gn *GameNode) IsTerminal() bool {
	return gn.turnType == GameOver
}

// Player implements cfr.GameTreeNode.
func (gn *GameNode) Player() int {
	return int(gn.player)
}

// NumChildren implements cfr.GameTreeNode.
func (gn *GameNode) NumChildren() int {
	gn.buildChildren()
	return len(gn.children)
}

// GetChild implements cfr.GameTreeNode.
func (gn *GameNode) GetChild(i int) cfr.GameTreeNode {
	gn.buildChildren()
	return &gn.children[i]
}

// GetChildProbability implements cfr.GameTreeNode.
func (gn *GameNode) GetChildProbability(i int) float64 {
	if !gn.IsChance() {
		panic("cannot get child probability on non-chance node")
	}

	gn.buildChildren()
	return gn.probabilities[i]
}

// InfoSet implements cfr.GameTreeNode.
func (gn *GameNode) InfoSet(player int) string {
	var buf strings.Builder
	is := gn.state.GetInfoSet(gamestate.Player(gn.player))
	binary.Write(&buf, binary.LittleEndian, is)
	return buf.String()
}

// Utility implements cfr.GameTreeNode.
func (gn *GameNode) Utility(player int) float64 {
	if !gn.IsTerminal() {
		panic("cannot get the utility of a non-terminal node")
	}

	if int(gn.player) == player {
		return 1.0
	}

	return -1.0
}

func (gn *GameNode) GetHistory() []gamestate.Action {
	return gn.state.GetHistory()
}

// String implements fmt.Stringer.
func (gn *GameNode) String() string {
	return fmt.Sprintf("%v's turn to %v. State: %s", gn.player, gn.turnType, gn.state.String())
}

func (gn *GameNode) allocChildren(n int) {
	gn.children = gn.gnPool.alloc(n)
	for i := range gn.children {
		gn.children[i] = *gn
	}

	if gn.turnType.IsChance() {
		gn.probabilities = gn.fPool.alloc(n)
	} else {
		gn.probabilities = nil
	}
}

func (gn *GameNode) buildChildren() {
	if len(gn.children) > 0 {
		return
	}

	switch gn.turnType {
	case DrawCard:
		gn.buildDrawCardChildren(false)
	case DrawCardFromBottom:
		gn.buildDrawCardChildren(true)
	case Deal:
		gn.buildDealChildren()
	case PlayTurn:
		gn.buildPlayTurnChildren()
	case GiveCard:
		gn.buildGiveCardChildren()
	case MustDefuse:
		gn.buildMustDefuseChildren()
	case SeeTheFuture:
		gn.buildSeeTheFutureChildren()
	}
}

func (gn *GameNode) buildDealChildren() {
	gn.allocChildren(0)
	// Deal 4 cards to player 0.
	player0Deals := enumerateInitialDeals(cards.CoreDeck, cards.NewSet(), cards.Unknown, 4, nil)
	for _, p0Deal := range player0Deals {
		remainingCards := cards.CoreDeck
		remainingCards.RemoveAll(p0Deal)
		// Deal 4 cards to player 1.
		player1Deals := enumerateInitialDeals(remainingCards, cards.NewSet(), cards.Unknown, 4, nil)
		for _, p1Deal := range player1Deals {
			// Player0 always goes first.
			gn.children = append(gn.children, *gn)
			child := &gn.children[len(gn.children)-1]
			child.state = gamestate.New(p0Deal, p1Deal)
			makePlayTurnNode(child, gamestate.Player0, 1)
		}
	}

	// All deals are equally likely.
	gn.probabilities = gn.fPool.alloc(len(gn.children))
	p := 1.0 / float64(len(gn.probabilities))
	for i := 0; i < len(gn.probabilities); i++ {
		gn.probabilities[i] = p
	}
}

// Chance node where the given player is drawing a card from the draw pile.
// If fromBottom == true, then the player is drawing from the bottom of the pile.
func makeDrawCardNode(node *GameNode, player gamestate.Player, fromBottom bool, pendingTurns int) {
	tt := DrawCard
	if fromBottom {
		tt = DrawCardFromBottom
	}

	node.player = player
	node.turnType = tt
	node.pendingTurns = pendingTurns
}

func makePlayTurnNode(node *GameNode, player gamestate.Player, pendingTurns int) {
	if pendingTurns == 0 {
		// Player's turn is done, next player.
		player = nextPlayer(player)
		pendingTurns = 1
	}

	node.player = player
	node.turnType = PlayTurn
	node.pendingTurns = pendingTurns
}

func makeGiveCardNode(node *GameNode, player gamestate.Player, pendingTurns int) {
	node.player = player
	node.turnType = GiveCard
	node.pendingTurns = pendingTurns
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

func makeSeeTheFutureNode(node *GameNode, player gamestate.Player, pendingTurns int) {
	node.player = player
	node.turnType = SeeTheFuture
	node.pendingTurns = pendingTurns
}

func makeTerminalGameNode(node *GameNode, winner gamestate.Player) {
	node.player = winner
	node.turnType = GameOver
}

func (gn *GameNode) buildDrawCardChildren(fromBottom bool) {
	if gn.state.GetDrawPile().Len() == 0 {
		panic(fmt.Errorf("trying to draw card but no cards in draw pile! %+v", gn.state))
	}

	// Drawing a card ends one turn.
	newPendingTurns := gn.pendingTurns - 1

	var nextCardProbs map[cards.Card]float64
	if fromBottom {
		nextCardProbs = gn.state.BottomCardProbabilities()
	} else {
		nextCardProbs = gn.state.TopCardProbabilities()
	}

	gn.allocChildren(len(nextCardProbs))
	i := 0
	for card, p := range nextCardProbs {
		gn.buildNextDrawnCardNode(&gn.children[i], card, fromBottom, newPendingTurns)
		gn.probabilities[i] = p
		i++
	}
}

func (gn *GameNode) buildNextDrawnCardNode(node *GameNode, card cards.Card, fromBottom bool, newPendingTurns int) {
	position := 0
	if fromBottom {
		position = gn.state.GetDrawPile().Len() - 1
	}

	node.state.Apply(gamestate.Action{
		Player:             gn.player,
		Type:               gamestate.DrawCard,
		Card:               card,
		PositionInDrawPile: position,
	})
	if card == cards.ExplodingCat {
		if gn.state.HasDefuseCard(gn.player) {
			// Player has a defuse card, must play it.
			makeMustDefuseNode(node, gn.player, newPendingTurns)
		} else {
			// Player does not have a defuse card, end game with loss for them.
			winner := nextPlayer(gn.player)
			makeTerminalGameNode(node, winner)
		}
	} else {
		// Just a normal card, add it to player's hand and continue.
		makePlayTurnNode(node, gn.player, newPendingTurns)
	}
}

func (gn *GameNode) buildSeeTheFutureChildren() {
	// TODO(palpant): Eliminate the allocations in enumerateTopNCards here.
	cards, probs := enumerateTopNCards(gn.state, 3)
	gn.allocChildren(len(cards))
	for i, top3 := range cards {
		gn.probabilities[i] = probs[i]

		child := &gn.children[i]
		child.state.Apply(gamestate.Action{
			Player: gn.player,
			Type:   gamestate.SeeTheFuture,
			Cards:  top3,
		})
		makePlayTurnNode(child, gn.player, gn.pendingTurns)
	}
}

func (gn *GameNode) buildPlayTurnChildren() {
	hand := gn.state.GetPlayerHand(gn.player)
	gn.allocChildren(hand.Len() + 1)
	// Play one of the cards in our hand.
	i := 0
	hand.CountsIter(func(card cards.Card, count uint8) {
		child := &gn.children[i]
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Updating opponent's view of the world to reflect played card,
		//   3) Updating game state based on action in card.
		child.state.Apply(gamestate.Action{
			Player: gn.player,
			Type:   gamestate.PlayCard,
			Card:   card,
		})

		switch card {
		case cards.Defuse:
			// No action besides losing card.
			makePlayTurnNode(child, gn.player, gn.pendingTurns)
		case cards.Skip:
			// Ends our current turn (without drawing a card).
			makePlayTurnNode(child, gn.player, gn.pendingTurns-1)
		case cards.Slap1x:
			// Ends our turn (and all pending turns). Goes to next player with
			// any pending turns + 1.
			if gn.state.LastActionWasSlap() {
				makePlayTurnNode(child, nextPlayer(gn.player), gn.pendingTurns+1)
			} else {
				makePlayTurnNode(child, nextPlayer(gn.player), 1)
			}
		case cards.Slap2x:
			// Ends our turn (and all pending turns). Goes to next player with
			// any pending turns + 2.
			if gn.state.LastActionWasSlap() {
				makePlayTurnNode(child, nextPlayer(gn.player), gn.pendingTurns+2)
			} else {
				makePlayTurnNode(child, nextPlayer(gn.player), 2)
			}
		case cards.SeeTheFuture:
			// We see the top 3 cards in the draw pile.
			makeSeeTheFutureNode(child, gn.player, gn.pendingTurns)
		case cards.Shuffle:
			makePlayTurnNode(child, gn.player, gn.pendingTurns)
		case cards.DrawFromTheBottom:
			makeDrawCardNode(child, gn.player, true, gn.pendingTurns)
		case cards.Cat:
			if child.state.GetPlayerHand(1-gn.player).Len() == 0 {
				// Other player has no cards in their hand, this was a no-op.
				makePlayTurnNode(child, gn.player, gn.pendingTurns)
			} else {
				// Other player must give us a card.
				makeGiveCardNode(child, nextPlayer(gn.player), gn.pendingTurns)
			}
		default:
			panic(fmt.Errorf("Player playing unsupported %v card", card))
		}

		i++
	})

	gn.children = gn.children[:i+1]
	// End our turn by drawing a card.
	lastChild := &gn.children[i]
	makeDrawCardNode(lastChild, gn.player, false, gn.pendingTurns)
}

func (gn *GameNode) buildGiveCardChildren() {
	hand := gn.state.GetPlayerHand(gn.player)
	gn.allocChildren(hand.Len())
	i := 0
	hand.CountsIter(func(card cards.Card, count uint8) {
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Adding card to opponent's hand,
		//   3) Returning to opponent's turn.
		child := &gn.children[i]
		child.state.Apply(gamestate.Action{
			Player: gn.player,
			Type:   gamestate.GiveCard,
			Card:   card,
		})

		// Game play returns to other player (with the given card in their hand).
		makePlayTurnNode(child, nextPlayer(gn.player), gn.pendingTurns)

		i++
	})

	gn.children = gn.children[:i]
}

func (gn *GameNode) buildMustDefuseChildren() {
	nCardsInDrawPile := int(gn.state.GetDrawPile().Len())
	nOptions := min(nCardsInDrawPile, 5)
	gn.allocChildren(nOptions + 1)
	for i := 0; i < nOptions; i++ {
		child := &gn.children[i]
		child.state.Apply(gamestate.Action{
			Player:             gn.player,
			Type:               gamestate.InsertExplodingCat,
			PositionInDrawPile: i,
		})

		// Defusing the exploding cat ends a turn.
		makePlayTurnNode(child, gn.player, gn.pendingTurns-1)
	}

	// Place exploding cat on the bottom of the draw pile.
	if nCardsInDrawPile > 5 {
		bottom := gn.state.GetDrawPile().Len()
		child := &gn.children[len(gn.children)-1]
		child.state.Apply(gamestate.Action{
			Player:             gn.player,
			Type:               gamestate.InsertExplodingCat,
			PositionInDrawPile: bottom,
		})

		// Defusing the exploding cat ends a turn.
		makePlayTurnNode(child, gn.player, gn.pendingTurns-1)
	} else {
		gn.children = gn.children[:len(gn.children)-1]
	}

	// FIXME: Place randomly?
}

func enumerateInitialDeals(available cards.Set, current cards.Set, card cards.Card, desired int, result []cards.Set) []cards.Set {
	if card > cards.Cat {
		return result
	}

	nRemaining := desired - current.Len()
	if nRemaining == 0 || nRemaining > available.Len() {
		return append(result, current)
	}

	count := int(available.CountOf(card))
	for i := 0; i <= min(count, nRemaining); i++ {
		current.AddN(card, i)
		available.RemoveN(card, i)
		result = enumerateInitialDeals(available, current, card+1, desired, result)
		current.RemoveN(card, i)
		available.AddN(card, i)
	}

	return result
}

func enumerateTopNCards(state gamestate.GameState, n int) ([][]cards.Card, []float64) {
	var result [][]cards.Card
	var resultProbs []float64

	nextCardProbs := state.TopCardProbabilities()
	drawPile := state.GetDrawPile()
	for card, p := range nextCardProbs {
		if n == 1 || drawPile.Len() == 1 {
			result = append(result, []cards.Card{card})
			resultProbs = append(resultProbs, p)
		} else { // Recurse to enumerate remaining n-1 cards.
			action := gamestate.Action{Player: gamestate.Player0, Type: gamestate.DrawCard, Card: card}
			remaining := state
			remaining.Apply(action)
			remainder, probs := enumerateTopNCards(remaining, n-1)
			for i, remain := range remainder {
				final := append([]cards.Card{card}, remain...)
				result = append(result, final)
				resultProbs = append(resultProbs, p*probs[i])
			}
		}
	}

	return result, resultProbs
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
