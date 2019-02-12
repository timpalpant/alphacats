package alphacats

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/internal/gamestate"
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
	return tt == DrawCard || tt == Deal || tt == SeeTheFuture
}

func (tt turnType) String() string {
	return turnTypeStr[tt]
}

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
	// If turnType is Chance, then these are the cumulative probabilities
	// of selecting each of the children. i.e. to select a child outcome,
	// draw a random number p \in (0, 1] and then select the first child with
	// cumulative probability >= p: children[cumulativeProbs >= p][0]
	cumulativeProbs []float64

	gnPool *gameNodeSlicePool
	fPool  *floatSlicePool
}

func NewGameTree() GameNode {
	gn := GameNode{
		player:   gamestate.Player0,
		turnType: Deal,
	}
	gn.BuildChildren()
	return gn
}

func (gn *GameNode) BuildChildren() {
	switch gn.turnType {
	case DrawCard:
		gn.buildDrawCardChildren(gn.state, gn.player, false, gn.pendingTurns)
	case DrawCardFromBottom:
		gn.buildDrawCardChildren(gn.state, gn.player, true, gn.pendingTurns)
	case Deal:
		gn.buildDealChildren()
	case PlayTurn:
		gn.buildPlayTurnChildren(gn.state, gn.player, gn.pendingTurns)
	case GiveCard:
		gn.buildGiveCardChildren(gn.state, gn.player, gn.pendingTurns)
	case MustDefuse:
		gn.buildMustDefuseChildren(gn.state, gn.player, gn.pendingTurns)
	case SeeTheFuture:
		gn.buildSeeTheFutureChildren(gn.state, gn.player, gn.pendingTurns)
	}
}

func (gn *GameNode) Clear() {
	gn.gnPool.free(gn.children)
	gn.children = nil
	gn.fPool.free(gn.cumulativeProbs)
	gn.cumulativeProbs = nil
}

func (gn *GameNode) IsTerminal() bool {
	return gn.turnType == GameOver
}

func (gn *GameNode) Winner() gamestate.Player {
	return gn.player
}

func (gn *GameNode) GetHistory() []gamestate.Action {
	return gn.state.GetHistory()
}

func (gn *GameNode) String() string {
	return fmt.Sprintf("%v's turn to %v. State: %s", gn.player, gn.turnType, gn.state.String())
}

func (gn *GameNode) NumChildren() int {
	return len(gn.children)
}

func (gn *GameNode) buildDealChildren() {
	gn.children = gn.gnPool.alloc()
	// Deal 4 cards to player 0.
	player0Deals := enumerateInitialDeals(cards.CoreDeck, cards.NewSet(), cards.Unknown, 4, nil)
	for _, p0Deal := range player0Deals {
		remainingCards := cards.CoreDeck
		remainingCards.RemoveAll(p0Deal)
		// Deal 4 cards to player 1.
		player1Deals := enumerateInitialDeals(remainingCards, cards.NewSet(), cards.Unknown, 4, nil)
		for _, p1Deal := range player1Deals {
			state := gamestate.New(p0Deal, p1Deal)
			// Player0 always goes first.
			node := gn.newPlayTurnNode(state, gamestate.Player0, 1)
			gn.children = append(gn.children, node)
		}
	}

	// All deals are equally likely.
	gn.cumulativeProbs = gn.fPool.alloc()
	for i := 0; i < len(gn.children); i++ {
		p := float64(i+1) / float64(len(gn.children))
		gn.cumulativeProbs = append(gn.cumulativeProbs, p)
	}
}

// Chance node where the given player is drawing a card from the draw pile.
// If fromBottom == true, then the player is drawing from the bottom of the pile.
func (gn *GameNode) newDrawCardNode(state gamestate.GameState, player gamestate.Player, fromBottom bool, pendingTurns int) GameNode {
	tt := DrawCard
	if fromBottom {
		tt = DrawCardFromBottom
	}

	return GameNode{
		state:        state,
		player:       player,
		turnType:     tt,
		pendingTurns: pendingTurns,
		gnPool:       gn.gnPool,
		fPool:        gn.fPool,
	}
}

func (gn *GameNode) newPlayTurnNode(state gamestate.GameState, player gamestate.Player, pendingTurns int) GameNode {
	if pendingTurns == 0 {
		// Player's turn is done, next player.
		player = nextPlayer(player)
		pendingTurns = 1
	}

	return GameNode{
		state:        state,
		player:       player,
		turnType:     PlayTurn,
		pendingTurns: pendingTurns,
		gnPool:       gn.gnPool,
		fPool:        gn.fPool,
	}
}

func (gn *GameNode) newGiveCardNode(state gamestate.GameState, player gamestate.Player, pendingTurns int) GameNode {
	return GameNode{
		state:        state,
		player:       player,
		turnType:     GiveCard,
		pendingTurns: pendingTurns,
		gnPool:       gn.gnPool,
		fPool:        gn.fPool,
	}
}

func (gn *GameNode) newMustDefuseNode(state gamestate.GameState, player gamestate.Player, pendingTurns int) GameNode {
	action := gamestate.Action{Player: player, Type: gamestate.PlayCard, Card: cards.Defuse}
	newState := gamestate.Apply(state, action)

	// Player may choose where to place exploding cat back in the draw pile.
	return GameNode{
		state:        newState,
		player:       player,
		turnType:     MustDefuse,
		pendingTurns: pendingTurns,
		gnPool:       gn.gnPool,
		fPool:        gn.fPool,
	}
}

func (gn *GameNode) newSeeTheFutureNode(state gamestate.GameState, player gamestate.Player, pendingTurns int) GameNode {
	return GameNode{
		state:        state,
		player:       player,
		turnType:     SeeTheFuture,
		pendingTurns: pendingTurns,
		gnPool:       gn.gnPool,
		fPool:        gn.fPool,
	}
}

func (gn *GameNode) newTerminalGameNode(state gamestate.GameState, winner gamestate.Player) GameNode {
	return GameNode{
		state:    state,
		turnType: GameOver,
		player:   winner,
		gnPool:   gn.gnPool,
		fPool:    gn.fPool,
	}
}

func (gn *GameNode) buildDrawCardChildren(state gamestate.GameState, player gamestate.Player, fromBottom bool, pendingTurns int) {
	if state.GetDrawPile().Len() == 0 {
		panic(fmt.Errorf("trying to draw card but no cards in draw pile! %+v", state))
	}

	// Drawing a card ends one turn.
	pendingTurns--

	var nextCardProbs map[cards.Card]float64
	if fromBottom {
		nextCardProbs = state.BottomCardProbabilities()
	} else {
		nextCardProbs = state.TopCardProbabilities()
	}

	gn.children = gn.gnPool.alloc()
	gn.cumulativeProbs = gn.fPool.alloc()
	cumProb := 0.0
	for card, p := range nextCardProbs {
		cumProb += p
		nextNode := gn.getNextDrawnCardNode(state, player, card, fromBottom, pendingTurns)
		gn.children = append(gn.children, nextNode)
		gn.cumulativeProbs = append(gn.cumulativeProbs, cumProb)
	}
}

func (gn *GameNode) getNextDrawnCardNode(state gamestate.GameState, player gamestate.Player, card cards.Card, fromBottom bool, pendingTurns int) GameNode {
	position := 0
	if fromBottom {
		position = state.GetDrawPile().Len() - 1
	}

	action := gamestate.Action{Player: player, Type: gamestate.DrawCard, Card: card, PositionInDrawPile: position}
	newState := gamestate.Apply(state, action)
	var nextNode GameNode
	if card == cards.ExplodingCat {
		if state.HasDefuseCard(player) {
			// Player has a defuse card, must play it.
			nextNode = gn.newMustDefuseNode(newState, player, pendingTurns)
		} else {
			// Player does not have a defuse card, end game with loss for them.
			winner := nextPlayer(player)
			nextNode = gn.newTerminalGameNode(newState, winner)
		}
	} else {
		// Just a normal card, add it to player's hand and continue.
		nextNode = gn.newPlayTurnNode(newState, player, pendingTurns)
	}

	return nextNode
}

func (gn *GameNode) buildSeeTheFutureChildren(state gamestate.GameState, player gamestate.Player, pendingTurns int) {
	cards, probs := enumerateTopNCards(state, 3)
	cumProb := 0.0
	gn.children = gn.gnPool.alloc()
	gn.cumulativeProbs = gn.fPool.alloc()
	for i, top3 := range cards {
		cumProb += probs[i]
		gn.cumulativeProbs = append(gn.cumulativeProbs, cumProb)

		action := gamestate.Action{Player: player, Type: gamestate.SeeTheFuture, Cards: top3}
		newState := gamestate.Apply(state, action)
		newNode := gn.newPlayTurnNode(newState, player, pendingTurns)
		gn.children = append(gn.children, newNode)
	}
}

func (gn *GameNode) buildPlayTurnChildren(state gamestate.GameState, player gamestate.Player, pendingTurns int) {
	gn.children = gn.gnPool.alloc()
	// Play one of the cards in our hand.
	state.GetPlayerHand(player).CountsIter(func(card cards.Card, count uint8) {
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Updating opponent's view of the world to reflect played card,
		//   3) Updating game state based on action in card.
		action := gamestate.Action{Player: player, Type: gamestate.PlayCard, Card: card}
		newGameState := gamestate.Apply(state, action)

		var nextNode GameNode
		switch card {
		case cards.Defuse:
			// No action besides losing card.
			nextNode = gn.newPlayTurnNode(newGameState, player, pendingTurns)
		case cards.Skip:
			// Ends our current turn (without drawing a card).
			nextNode = gn.newPlayTurnNode(newGameState, player, pendingTurns-1)
		case cards.Slap1x:
			// Ends our turn (and all pending turns). Goes to next player with
			// any pending turns + 1.
			// FIXME: This should look to see whether previous action was slap.
			if pendingTurns == 1 {
				nextNode = gn.newPlayTurnNode(newGameState, nextPlayer(player), 1)
			} else {
				nextNode = gn.newPlayTurnNode(newGameState, nextPlayer(player), pendingTurns+1)
			}
		case cards.Slap2x:
			// Ends our turn (and all pending turns). Goes to next player with
			// any pending turns + 2.
			// FIXME: This should look to see whether previous action was slap.
			if pendingTurns == 1 {
				nextNode = gn.newPlayTurnNode(newGameState, nextPlayer(player), 2)
			} else {
				nextNode = gn.newPlayTurnNode(newGameState, nextPlayer(player), pendingTurns+2)
			}
		case cards.SeeTheFuture:
			// We see the top 3 cards in the draw pile.
			nextNode = gn.newSeeTheFutureNode(newGameState, player, pendingTurns)
		case cards.Shuffle:
			nextNode = gn.newPlayTurnNode(newGameState, player, pendingTurns)
		case cards.DrawFromTheBottom:
			nextNode = gn.newDrawCardNode(newGameState, player, true, pendingTurns)
		case cards.Cat:
			if newGameState.GetPlayerHand(1-player).Len() == 0 {
				// Other player has no cards in their hand, this was a no-op.
				nextNode = gn.newPlayTurnNode(newGameState, player, pendingTurns)
			} else {
				// Other player must give us a card.
				nextNode = gn.newGiveCardNode(newGameState, nextPlayer(player), pendingTurns)
			}
		default:
			panic(fmt.Errorf("Player playing unsupported %v card", card))
		}

		gn.children = append(gn.children, nextNode)
	})

	// End our turn by drawing a card.
	nextNode := gn.newDrawCardNode(state, player, false, pendingTurns)
	gn.children = append(gn.children, nextNode)
}

func (gn *GameNode) buildGiveCardChildren(state gamestate.GameState, player gamestate.Player, pendingTurns int) {
	gn.children = gn.gnPool.alloc()
	state.GetPlayerHand(player).CountsIter(func(card cards.Card, count uint8) {
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Adding card to opponent's hand,
		//   3) Returning to opponent's turn.
		action := gamestate.Action{Player: player, Type: gamestate.GiveCard, Card: card}
		newState := gamestate.Apply(state, action)

		// Game play returns to other player (with the given card in their hand).
		nextNode := gn.newPlayTurnNode(newState, nextPlayer(player), pendingTurns)
		gn.children = append(gn.children, nextNode)
	})
}

func (gn *GameNode) buildMustDefuseChildren(state gamestate.GameState, player gamestate.Player, pendingTurns int) {
	gn.children = gn.gnPool.alloc()
	nCardsInDrawPile := int(state.GetDrawPile().Len())
	nOptions := min(nCardsInDrawPile, 5)
	for i := 0; i <= nOptions; i++ {
		action := gamestate.Action{
			Player:             player,
			Type:               gamestate.InsertExplodingCat,
			PositionInDrawPile: i,
		}
		newState := gamestate.Apply(state, action)
		// Defusing the exploding cat ends a turn.
		nextNode := gn.newPlayTurnNode(newState, player, pendingTurns-1)
		gn.children = append(gn.children, nextNode)
	}

	// Place exploding cat on the bottom of the draw pile.
	if nCardsInDrawPile > 5 {
		bottom := state.GetDrawPile().Len()
		action := gamestate.Action{
			Player:             player,
			Type:               gamestate.InsertExplodingCat,
			PositionInDrawPile: bottom,
		}
		newState := gamestate.Apply(state, action)
		// Defusing the exploding cat ends a turn.
		nextNode := gn.newPlayTurnNode(newState, player, pendingTurns-1)
		gn.children = append(gn.children, nextNode)
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
			remaining := gamestate.Apply(state, action)
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
