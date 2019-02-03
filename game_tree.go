package alphacats

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"

	"github.com/golang/glog"
)

// GameNode represents a state of play in the extensive-form game tree.
type GameNode struct {
	state        GameState
	player       Player
	turnType     TurnType
	fromBottom   bool
	pendingTurns int

	children []*GameNode
	// If turnType is Chance, then these are the cumulative probabilities
	// of selecting each of the children. i.e. to select a child outcome,
	// draw a random number p \in (0, 1] and then select the first child with
	// cumulative probability >= p: children[cumulativeProbs >= p][0]
	cumulativeProbs []float64
}

func (gn *GameNode) String() string {
	return fmt.Sprintf("%v - %s: %+v", gn.player, gn.turnType, gn.state)
}

func (gn *GameNode) buildChildren() {
	glog.V(2).Info("Building node children")
	switch gn.turnType {
	case DrawCard:
		children, probs := buildDrawCardChildren(gn.state, gn.player, gn.fromBottom, gn.pendingTurns)
		gn.children = children
		gn.cumulativeProbs = probs
	case Deal:
		gn.children = buildDealChildren()
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

func NewGameTree() *GameNode {
	children := buildDealChildren()
	probs := make([]float64, len(children))
	// FIXME: Should not be a uniform distribution!
	for i := 0; i < len(probs); i++ {
		probs[i] = float64(i+1) / float64(len(probs))
	}

	return &GameNode{
		player:          Player0,
		turnType:        Deal,
		children:        children,
		cumulativeProbs: probs,
	}
}

func buildDealChildren() []*GameNode {
	result := make([]*GameNode, 0)
	// Deal 4 cards to player 0.
	player0Deals := enumerateInitialDeals(cards.CoreDeck, cards.NewSet(), cards.Unknown, 4, nil)
	glog.V(1).Infof("Enumerated %d initial deals for Player0", len(player0Deals))
	for _, p0Deal := range player0Deals {
		remainingCards := cards.CoreDeck
		remainingCards.RemoveAll(p0Deal)
		// Deal 4 cards to player 1.
		player1Deals := enumerateInitialDeals(remainingCards, cards.NewSet(), cards.Unknown, 4, nil)
		glog.V(1).Infof("Enumerated %d initial deals for Player1", len(player1Deals))
		for _, p1Deal := range player1Deals {
			glog.V(2).Infof("P0 deal: %v, P1 deal: %v", p0Deal, p1Deal)
			gameState := buildInitialGameState(p0Deal, p1Deal)
			// Player0 always goes first.
			node := newPlayTurnNode(gameState, Player0, 1)
			result = append(result, node)
		}
	}

	glog.Infof("Built %d initial game states", len(result))
	return result
}

func buildInitialGameState(player1Deal, player2Deal cards.Set) GameState {
	remainingCards := cards.CoreDeck
	remainingCards.RemoveAll(player1Deal)
	remainingCards.RemoveAll(player2Deal)
	remainingCards.Add(cards.ExplodingCat)
	return GameState{
		DrawPile:    remainingCards,
		Player0Info: NewInfoSetFromInitialDeal(player1Deal),
		Player1Info: NewInfoSetFromInitialDeal(player2Deal),
	}
}

// Chance node where the given player is drawing a card from the draw pile.
// If fromBottom == true, then the player is drawing from the bottom of the pile.
func newDrawCardNode(state GameState, player Player, fromBottom bool, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building draw card node: player = %v, from bottom = %v, pending turns = %v",
		player, fromBottom, pendingTurns)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	return &GameNode{
		state:        state,
		player:       player,
		turnType:     DrawCard,
		fromBottom:   fromBottom,
		pendingTurns: pendingTurns,
	}
}

func newPlayTurnNode(state GameState, player Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building play turn node: player = %v, pending turns = %v",
		player, pendingTurns)

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

func newGiveCardNode(state GameState, player Player, pendingTurns int) *GameNode {
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

func newMustDefuseNode(state GameState, player Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building must defuse node: player = %v, pending turns = %v",
		player, pendingTurns)

	newState := state
	newState.InfoSet(player).PlayCard(cards.Defuse)
	newState.InfoSet(1 - player).OpponentPlayedCard(cards.Defuse)
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

func newSeeTheFutureNode(state GameState, player Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building see the future node: player = %v, pending turns = %v",
		player, pendingTurns)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	return &GameNode{
		state:        state,
		player:       player,
		pendingTurns: pendingTurns,
	}
}

func buildDrawCardChildren(state GameState, player Player, fromBottom bool, pendingTurns int) ([]*GameNode, []float64) {
	if state.DrawPile.Len() == 0 {
		panic(fmt.Errorf("trying to draw card but no cards in draw pile! %+v", state))
	}

	var result []*GameNode
	var probs []float64
	// Drawing a card ends one turn.
	pendingTurns--

	// Check if card being drawn is known. If it is then this node is
	// deterministic, return it.
	if fromBottom && bottomCardIsKnown(state) {
		bottom := state.DrawPile.Len() - 1
		card := state.FixedDrawPileCards().NthCard(bottom)
		nextNode := getNextDrawnCardNode(state, player, card, fromBottom, pendingTurns)
		result = append(result, nextNode)
		probs = append(probs, 1.0)
		return result, probs
	} else if !fromBottom && topCardIsKnown(state) {
		card := state.FixedDrawPileCards().NthCard(0)
		nextNode := getNextDrawnCardNode(state, player, card, fromBottom, pendingTurns)
		result = append(result, nextNode)
		probs = append(probs, 1.0)
		return result, probs
	}

	// Else: choose a card randomly according to distribution.
	cumProb := 0.0
	for _, card := range state.DrawPile.Distinct() {
		p := float64(state.DrawPile.CountOf(card)) / float64(state.DrawPile.Len())
		cumProb += p
		nextNode := getNextDrawnCardNode(state, player, card, fromBottom, pendingTurns)
		result = append(result, nextNode)
		probs = append(probs, cumProb)
	}

	glog.V(3).Infof("Built %d draw card children with probs: %v", len(result), probs)
	return result, probs
}

func getNextDrawnCardNode(state GameState, player Player, card cards.Card, fromBottom bool, pendingTurns int) *GameNode {
	newState := playerDrewCard(state, player, card, fromBottom)
	var nextNode *GameNode
	if card == cards.ExplodingCat {
		if playerHasDefuseCard(state, player) {
			// Player has a defuse card, must play it.
			nextNode = newMustDefuseNode(newState, player, pendingTurns)
		} else {
			// Player does not have a defuse card, end game with loss for them.
			winner := nextPlayer(player)
			glog.V(3).Infof("Reached terminal node with player %v win", winner)
			nextNode = &GameNode{turnType: GameOver, player: winner}
		}
	} else {
		// Just a normal card, add it to player's hand and continue.
		nextNode = newPlayTurnNode(newState, player, pendingTurns)
	}

	return nextNode
}

func playerDrewCard(state GameState, player Player, card cards.Card, fromBottom bool) GameState {
	newState := state
	// Pop card from the draw pile.
	newState.DrawPile.Remove(card)
	// Add to player's hand.
	newState.InfoSet(player).DrawCard(card, fromBottom)
	newState.InfoSet(1-player).OpponentDrewCard(card, fromBottom)
	return newState
}

func buildSeeTheFutureChildren(state GameState, player Player, pendingTurns int) ([]*GameNode, []float64) {
	var result []*GameNode
	var cumulativeProbs []float64

	cards, probs := enumerateTopNCards(state.DrawPile, state.FixedDrawPileCards(), 3)
	var cumProb float64
	for i, topN := range cards {
		cumProb += probs[i]
		cumulativeProbs = append(cumulativeProbs, cumProb)

		newState := state
		newState.InfoSet(player).SeeTopCards(topN)
		newNode := newPlayTurnNode(newState, player, pendingTurns)
		result = append(result, newNode)
	}

	glog.V(3).Infof("Built %d see the future children", len(result))
	return result, cumulativeProbs
}

func enumerateTopNCards(drawPile cards.Set, fixed cards.Stack, n int) ([][]cards.Card, []float64) {
	var result [][]cards.Card
	var resultProbs []float64

	nextCardProbabilities := make(map[cards.Card]float64)
	topCard := fixed.NthCard(0)
	if topCard != cards.Unknown { // Card is fixed
		nextCardProbabilities[topCard] = 1.0
	} else {
		for _, card := range drawPile.Distinct() {
			p := float64(drawPile.CountOf(card)) / float64(drawPile.Len())
			nextCardProbabilities[card] = p
		}
	}

	for card, p := range nextCardProbabilities {
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

func topCardIsKnown(state GameState) bool {
	return state.FixedDrawPileCards().NthCard(0) != cards.Unknown
}

func bottomCardIsKnown(state GameState) bool {
	bottom := state.DrawPile.Len() - 1
	return state.FixedDrawPileCards().NthCard(bottom) != cards.Unknown
}

func playerHasDefuseCard(state GameState, player Player) bool {
	return state.InfoSet(player).OurHand.CountOf(cards.Defuse) > 0
}

func buildPlayTurnChildren(state GameState, player Player, pendingTurns int) []*GameNode {
	// Choose whether to play a card or draw.
	cardChoices := state.InfoSet(player).OurHand.Distinct()
	result := make([]*GameNode, 0, len(cardChoices)+1)

	// Play one of the cards in our hand.
	for _, card := range cardChoices {
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Updating opponent's view of the world to reflect played card,
		//   3) Updating game state based on action in card.
		newGameState := state
		newGameState.InfoSet(player).PlayCard(card)
		newGameState.InfoSet(1 - player).OpponentPlayedCard(card)

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
			newGameState = shuffleDrawPile(newGameState)
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

func shuffleDrawPile(state GameState) GameState {
	result := state
	result.Player0Info.KnownDrawPileCards = cards.NewStack()
	result.Player1Info.KnownDrawPileCards = cards.NewStack()
	return result
}

func buildGiveCardChildren(state GameState, player Player, pendingTurns int) []*GameNode {
	cardChoices := state.GetPlayerHand(player).Distinct()
	result := make([]*GameNode, 0, len(cardChoices))
	for _, card := range cardChoices {
		glog.V(4).Infof("Player %v giving card %v", player, card)
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Adding card to opponent's hand,
		//   3) Returning to opponent's turn.
		newGameState := state

		ourInfo := newGameState.InfoSet(player)
		ourInfo.OurHand.Remove(card)
		ourInfo.OpponentHand.Add(card)

		opponentInfo := newGameState.InfoSet(1 - player)
		opponentInfo.OurHand.Add(card)
		if opponentInfo.OpponentHand.CountOf(card) > 0 {
			// If opponent already knew we had one of these cards
			opponentInfo.OpponentHand.Remove(card)
		} else {
			// Otherwise it was one of the Unknown cards in our hand.
			opponentInfo.OpponentHand.Remove(cards.Unknown)
		}

		// Game play returns to other player (with the given card in their hand).
		nextNode := newPlayTurnNode(newGameState, nextPlayer(player), pendingTurns)
		result = append(result, nextNode)
	}

	glog.V(3).Infof("Built %d give card children", len(result))
	return result
}

func buildMustDefuseChildren(state GameState, player Player, pendingTurns int) []*GameNode {
	result := make([]*GameNode, 0)
	nCardsInDrawPile := int(state.DrawPile.Len())
	nOptions := min(nCardsInDrawPile, 5)
	for i := 0; i <= nOptions; i++ {
		newState := insertExplodingCat(state, player, i)
		// Defusing the exploding cat ends a turn.
		nextNode := newPlayTurnNode(newState, player, pendingTurns-1)
		result = append(result, nextNode)
	}

	// Place exploding cat on the bottom of the draw pile.
	if nCardsInDrawPile > 5 {
		bottom := state.DrawPile.Len()
		newState := insertExplodingCat(state, player, bottom)
		// Defusing the exploding cat ends a turn.
		nextNode := newPlayTurnNode(newState, player, pendingTurns-1)
		result = append(result, nextNode)
	}

	// TODO: Place randomly?
	glog.V(3).Infof("Built %d defuse children", len(result))
	return result
}

func insertExplodingCat(state GameState, player Player, position int) GameState {
	newState := state
	// Place exploding cat card in the Nth position in draw pile.
	newState.DrawPile.Add(cards.ExplodingCat)
	newState.InfoSet(player).OurHand.Remove(cards.ExplodingCat)
	newState.InfoSet(player).KnownDrawPileCards.InsertCard(cards.ExplodingCat, position)
	newState.InfoSet(1 - player).OpponentHand.Remove(cards.ExplodingCat)
	// FIXME: Known draw pile cards is not completely reset,
	// just reset until we get to the cat. Player's knowledge beneath the
	// insertion should be retained!
	newState.InfoSet(1 - player).KnownDrawPileCards = cards.NewStack()
	return newState
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

func min(i, j int) int {
	if i < j {
		return i
	}

	return j
}
