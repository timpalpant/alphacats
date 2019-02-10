package alphacats

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"

	"github.com/golang/glog"
)

// turnType represents the kind of turn at a given point in the game.
type TurnType uint8

const (
	_ TurnType = iota
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
	"Deal",
	"PlayTurn",
	"GiveCard",
	"MustDefuse",
	"SeeTheFuture",
	"GameOver",
}

func (tt TurnType) IsChance() bool {
	return tt == DrawCard || tt == Deal || tt == SeeTheFuture
}

func (tt TurnType) String() string {
	return turnTypeStr[tt]
}

// GameNode represents a state of play in the extensive-form game tree.
type GameNode struct {
	state    gamestate.GameState
	player   gamestate.Player
	turnType TurnType
	// pendingTurns is the number of turns the player has outstanding
	// to play. In general this will be 1, except when Slap cards are played.
	pendingTurns int

	// children are the possible next states in the game.
	// Which child is realized will depend on chance or a player's action.
	children []*GameNode
	// If turnType is Chance, then these are the cumulative probabilities
	// of selecting each of the children. i.e. to select a child outcome,
	// draw a random number p \in (0, 1] and then select the first child with
	// cumulative probability >= p: children[cumulativeProbs >= p][0]
	cumulativeProbs []float64
}

func NewGameTree() *GameNode {
	gn := &GameNode{
		player:   gamestate.Player0,
		turnType: Deal,
	}
	gn.buildChildren()
	return gn
}

func (gn *GameNode) GetHistory() []gamestate.Action {
	return gn.state.GetHistory()
}

func (gn *GameNode) String() string {
	return fmt.Sprintf("%v's turn to %v", gn.player, gn.turnType)
}

func (gn *GameNode) buildChildren() {
	glog.V(2).Info("Building node children")
	switch gn.turnType {
	case DrawCard:
		children, probs := buildDrawCardChildren(gn.state, gn.player, false, gn.pendingTurns)
		gn.children = children
		gn.cumulativeProbs = probs
	case DrawCardFromBottom:
		children, probs := buildDrawCardChildren(gn.state, gn.player, true, gn.pendingTurns)
		gn.children = children
		gn.cumulativeProbs = probs
	case Deal:
		children, probs := buildDealChildren()
		gn.children = children
		gn.cumulativeProbs = probs
	case PlayTurn:
		gn.children = buildPlayTurnChildren(gn.state, gn.player, gn.pendingTurns)
	case GiveCard:
		gn.children = buildGiveCardChildren(gn.state, gn.player, gn.pendingTurns)
	case MustDefuse:
		gn.children = buildMustDefuseChildren(gn.state, gn.player, gn.pendingTurns)
	case SeeTheFuture:
		children, probs := buildSeeTheFutureChildren(gn.state, gn.player, gn.pendingTurns)
		gn.children = children
		gn.cumulativeProbs = probs
	}
}

func (gn *GameNode) NumChildren() int {
	return len(gn.children)
}

func buildDealChildren() ([]*GameNode, []float64) {
	result := make([]*GameNode, 0)
	// Deal 4 cards to player 0.
	player0Deals := enumerateInitialDeals(cards.CoreDeck, cards.NewSet(), cards.Unknown, 4, nil)
	glog.V(1).Infof("Enumerated %d initial deals for Player0", len(player0Deals))
	for _, p0Deal := range player0Deals {
		remainingCards := cards.CoreDeck
		remainingCards.RemoveAll(p0Deal)
		// Deal 4 cards to player 1.
		player1Deals := enumerateInitialDeals(remainingCards, cards.NewSet(), cards.Unknown, 4, nil)
		glog.V(2).Infof("Enumerated %d initial deals for Player1", len(player1Deals))
		for _, p1Deal := range player1Deals {
			glog.V(3).Infof("P0 deal: %v, P1 deal: %v", p0Deal, p1Deal)
			state := gamestate.New(p0Deal, p1Deal)
			// Player0 always goes first.
			node := newPlayTurnNode(state, gamestate.Player0, 1)
			result = append(result, node)
		}
	}

	glog.V(1).Infof("Built %d initial game states", len(result))

	// All deals are equally likely.
	probs := make([]float64, len(result))
	for i := 0; i < len(probs); i++ {
		probs[i] = float64(i+1) / float64(len(probs))
	}

	return result, probs
}

// Chance node where the given player is drawing a card from the draw pile.
// If fromBottom == true, then the player is drawing from the bottom of the pile.
func newDrawCardNode(state gamestate.GameState, player gamestate.Player, fromBottom bool, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building draw card node: player = %v, from bottom = %v, pending turns = %v",
		player, fromBottom, pendingTurns)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	tt := DrawCard
	if fromBottom {
		tt = DrawCardFromBottom
	}

	return &GameNode{
		state:        state,
		player:       player,
		turnType:     tt,
		pendingTurns: pendingTurns,
	}
}

func newPlayTurnNode(state gamestate.GameState, player gamestate.Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building play turn node: player = %v, pending turns = %v",
		player, pendingTurns)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	if pendingTurns == 0 {
		// Player's turn is done, next player.
		player = nextPlayer(player)
		pendingTurns = 1
	}

	return &GameNode{
		state:        state,
		player:       player,
		turnType:     PlayTurn,
		pendingTurns: pendingTurns,
	}
}

func newGiveCardNode(state gamestate.GameState, player gamestate.Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building give card node: player = %v, pending turns = %v",
		player, pendingTurns)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	return &GameNode{
		state:        state,
		player:       player,
		turnType:     GiveCard,
		pendingTurns: pendingTurns,
	}
}

func newMustDefuseNode(state gamestate.GameState, player gamestate.Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building must defuse node: player = %v, pending turns = %v",
		player, pendingTurns)

	action := gamestate.Action{Player: player, Type: gamestate.PlayCard, Card: cards.Defuse}
	newState := gamestate.Apply(state, action)
	if err := newState.Validate(); err != nil {
		glog.Errorf("State: %+v", newState)
		panic(err)
	}

	// Player may choose where to place exploding cat back in the draw pile.
	return &GameNode{
		state:        newState,
		player:       player,
		turnType:     MustDefuse,
		pendingTurns: pendingTurns,
	}
}

func newSeeTheFutureNode(state gamestate.GameState, player gamestate.Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building see the future node: player = %v, pending turns = %v",
		player, pendingTurns)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	return &GameNode{
		state:        state,
		player:       player,
		turnType:     SeeTheFuture,
		pendingTurns: pendingTurns,
	}
}

func newTerminalGameNode(state gamestate.GameState, winner gamestate.Player) *GameNode {
	return &GameNode{
		state:    state,
		turnType: GameOver,
		player:   winner,
	}
}

func buildDrawCardChildren(state gamestate.GameState, player gamestate.Player, fromBottom bool, pendingTurns int) ([]*GameNode, []float64) {
	if state.GetDrawPile().Len() == 0 {
		panic(fmt.Errorf("trying to draw card but no cards in draw pile! %+v", state))
	}

	// Drawing a card ends one turn.
	pendingTurns--

	var result []*GameNode
	var probs []float64
	cumProb := 0.0
	nextCards := nextCardProbabilities(state.GetDrawPile(), state.FixedDrawPileCards(), fromBottom)
	for card, p := range nextCards {
		cumProb += p
		nextNode := getNextDrawnCardNode(state, player, card, fromBottom, pendingTurns)
		result = append(result, nextNode)
		probs = append(probs, cumProb)
	}

	glog.V(3).Infof("Built %d draw card children with probs: %v", len(result), probs)
	return result, probs
}

func nextCardProbabilities(drawPile cards.Set, fixed cards.Stack, fromBottom bool) map[cards.Card]float64 {
	result := make(map[cards.Card]float64)

	bottom := drawPile.Len() - 1
	bottomCard := fixed.NthCard(bottom)
	topCard := fixed.NthCard(0)
	if fromBottom && bottomCard != cards.Unknown {
		result[bottomCard] = 1.0
		return result
	} else if !fromBottom && topCard != cards.Unknown {
		result[topCard] = 1.0
		return result
	}

	// Note: We need to exclude any cards whose identity is already fixed in a
	// position known not to be the top (bottom) card.
	// Loop over all fixed cards NOT in the location we are drawing.
	var start, end int
	if fromBottom { // If drawing from the bottom.
		start = 0
		end = drawPile.Len() - 1
	} else { // If drawing from the top.
		start = 1
		end = drawPile.Len()
	}

	candidates := drawPile
	for i := start; i < end; i++ {
		if known := fixed.NthCard(i); known != cards.Unknown {
			candidates.Remove(known)
		}
	}

	counts := candidates.Counts()
	nCandidates := float64(candidates.Len())
	for card, count := range counts {
		p := float64(count) / nCandidates
		result[card] = p
	}

	return result
}

func getNextDrawnCardNode(state gamestate.GameState, player gamestate.Player, card cards.Card, fromBottom bool, pendingTurns int) *GameNode {
	position := 0
	if fromBottom {
		position = state.GetDrawPile().Len() - 1
	}

	action := gamestate.Action{Player: player, Type: gamestate.DrawCard, Card: card, PositionInDrawPile: position}
	newState := gamestate.Apply(state, action)
	var nextNode *GameNode
	if card == cards.ExplodingCat {
		if state.HasDefuseCard(player) {
			// Player has a defuse card, must play it.
			nextNode = newMustDefuseNode(newState, player, pendingTurns)
		} else {
			// Player does not have a defuse card, end game with loss for them.
			winner := nextPlayer(player)
			glog.V(3).Infof("Reached terminal node with player %v win", winner)
			nextNode = newTerminalGameNode(newState, player)
		}
	} else {
		// Just a normal card, add it to player's hand and continue.
		nextNode = newPlayTurnNode(newState, player, pendingTurns)
	}

	return nextNode
}

func buildSeeTheFutureChildren(state gamestate.GameState, player gamestate.Player, pendingTurns int) ([]*GameNode, []float64) {
	var result []*GameNode
	var cumulativeProbs []float64

	cards, probs := enumerateTopNCards(state.GetDrawPile(), state.FixedDrawPileCards(), 3)
	var cumProb float64
	for i, top3 := range cards {
		cumProb += probs[i]
		cumulativeProbs = append(cumulativeProbs, cumProb)

		action := gamestate.Action{Player: player, Type: gamestate.SeeTheFuture, Cards: top3}
		newState := gamestate.Apply(state, action)
		// FIXME: How do we represent the second-order knowledge for 1-player
		// that player now knows the top 3 cards?
		newNode := newPlayTurnNode(newState, player, pendingTurns)
		result = append(result, newNode)
	}

	glog.V(3).Infof("Built %d see the future children", len(result))
	return result, cumulativeProbs
}

func enumerateTopNCards(drawPile cards.Set, fixed cards.Stack, n int) ([][]cards.Card, []float64) {
	var result [][]cards.Card
	var resultProbs []float64

	nextCards := nextCardProbabilities(drawPile, fixed, false)
	for card, p := range nextCards {
		if n == 1 || drawPile.Len() == 1 {
			result = append(result, []cards.Card{card})
			resultProbs = append(resultProbs, p)
		} else { // Recurse to enumerate remaining n-1 cards.
			remainingDrawPile := drawPile
			remainingDrawPile.Remove(card)
			remainingFixed := fixed
			remainingFixed.RemoveCard(0)
			remainder, probs := enumerateTopNCards(remainingDrawPile, remainingFixed, n-1)
			for i, remain := range remainder {
				final := append([]cards.Card{card}, remain...)
				result = append(result, final)
				resultProbs = append(resultProbs, p*probs[i])
			}
		}
	}

	return result, resultProbs
}

func buildPlayTurnChildren(state gamestate.GameState, player gamestate.Player, pendingTurns int) []*GameNode {
	// Choose whether to play a card or draw.
	cardChoices := state.GetPlayerHand(player).Distinct()
	result := make([]*GameNode, 0, len(cardChoices)+1)

	// Play one of the cards in our hand.
	for _, card := range cardChoices {
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Updating opponent's view of the world to reflect played card,
		//   3) Updating game state based on action in card.
		action := gamestate.Action{Player: player, Type: gamestate.PlayCard, Card: card}
		newGameState := gamestate.Apply(state, action)

		var nextNode *GameNode
		glog.V(3).Infof("Player %v playing %v card", player, card)
		switch card {
		case cards.Defuse:
			// No action besides losing card.
			nextNode = newPlayTurnNode(newGameState, player, pendingTurns)
		case cards.Skip:
			// Ends our current turn (without drawing a card).
			nextNode = newPlayTurnNode(newGameState, player, pendingTurns-1)
		case cards.Slap1x:
			// Ends our turn (and all pending turns). Goes to next player with
			// any pending turns + 1.
			if pendingTurns == 1 {
				nextNode = newPlayTurnNode(newGameState, nextPlayer(player), 1)
			} else {
				nextNode = newPlayTurnNode(newGameState, nextPlayer(player), pendingTurns+1)
			}
		case cards.Slap2x:
			// Ends our turn (and all pending turns). Goes to next player with
			// any pending turns + 2.
			if pendingTurns == 1 {
				nextNode = newPlayTurnNode(newGameState, nextPlayer(player), 2)
			} else {
				nextNode = newPlayTurnNode(newGameState, nextPlayer(player), pendingTurns+2)
			}
		case cards.SeeTheFuture:
			// We see the top 3 cards in the draw pile.
			nextNode = newSeeTheFutureNode(newGameState, player, pendingTurns)
		case cards.Shuffle:
			newGameState = gamestate.ShuffleDrawPile(newGameState)
			nextNode = newPlayTurnNode(newGameState, player, pendingTurns)
		case cards.DrawFromTheBottom:
			nextNode = newDrawCardNode(newGameState, player, true, pendingTurns)
		case cards.Cat:
			if newGameState.GetPlayerHand(1-player).Len() == 0 {
				// Other player has no cards in their hand, this was a no-op.
				nextNode = newPlayTurnNode(newGameState, player, pendingTurns)
			} else {
				// Other player must give us a card.
				nextNode = newGiveCardNode(newGameState, nextPlayer(player), pendingTurns)
			}
		default:
			panic(fmt.Errorf("Player playing unsupported %v card", card))
		}

		result = append(result, nextNode)
	}

	// End our turn by drawing a card.
	nextNode := newDrawCardNode(state, player, false, pendingTurns)
	result = append(result, nextNode)

	glog.V(3).Infof("Built %d play turn children", len(result))
	return result
}

func buildGiveCardChildren(state gamestate.GameState, player gamestate.Player, pendingTurns int) []*GameNode {
	cardChoices := state.GetPlayerHand(player).Distinct()
	result := make([]*GameNode, 0, len(cardChoices))
	for _, card := range cardChoices {
		glog.V(4).Infof("Player %v giving card %v", player, card)
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Adding card to opponent's hand,
		//   3) Returning to opponent's turn.
		newGameState := gamestate.GiveCard(state, player, card)

		// Game play returns to other player (with the given card in their hand).
		nextNode := newPlayTurnNode(newGameState, nextPlayer(player), pendingTurns)
		result = append(result, nextNode)
	}

	glog.V(3).Infof("Built %d give card children", len(result))
	return result
}

func buildMustDefuseChildren(state gamestate.GameState, player gamestate.Player, pendingTurns int) []*GameNode {
	result := make([]*GameNode, 0)
	nCardsInDrawPile := int(state.GetDrawPile().Len())
	nOptions := min(nCardsInDrawPile, 5)
	for i := 0; i <= nOptions; i++ {
		newState := gamestate.InsertExplodingCat(state, player, i)
		// Defusing the exploding cat ends a turn.
		nextNode := newPlayTurnNode(newState, player, pendingTurns-1)
		result = append(result, nextNode)
	}

	// Place exploding cat on the bottom of the draw pile.
	if nCardsInDrawPile > 5 {
		bottom := state.GetDrawPile().Len()
		newState := gamestate.InsertExplodingCat(state, player, bottom)
		// Defusing the exploding cat ends a turn.
		nextNode := newPlayTurnNode(newState, player, pendingTurns-1)
		result = append(result, nextNode)
	}

	// TODO: Place randomly?
	glog.V(3).Infof("Built %d defuse children", len(result))
	return result
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
