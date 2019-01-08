package alphacats

import (
	"fmt"
	"math/rand"

	"github.com/timpalpant/alphacats/cards"

	"github.com/golang/glog"
	"github.com/pkg/errors"
)

type Player uint8

const (
	Player0 Player = iota
	Player1
	Chance
)

func nextPlayer(p Player) Player {
	return 1 - p
}

// GameState represents the current state of the game.
type GameState struct {
	// Set of Cards remaining in the draw pile.
	DrawPile cards.Set
	// Cards in the draw pile whose position/identity has been fixed
	// because one of the players knows it (either because of a
	// SeeTheFuture card or replacing a card in the deck).
	FixedDrawPileCards cards.Stack
	// Info observable from the point of view of either player.
	Player0Info InfoSet
	Player1Info InfoSet
}

// Validate sanity checks the GameState to ensure we have maintained
// internal consistency in the game tree.
func (gs *GameState) Validate() error {
	if err := gs.Player0Info.Validate(); err != nil {
		return errors.Wrapf(err, "player %v info invalid", Player0)
	}

	if err := gs.Player1Info.Validate(); err != nil {
		return errors.Wrapf(err, "player %v info invalid", Player1)
	}

	// All player's should know the current number of cards in the draw pile.
	if gs.DrawPile.Len() != gs.Player0Info.NumCardsInDrawPile() {
		return fmt.Errorf("player %v thinks draw pile has %d cards, actually %d",
			Player0, gs.Player0Info.NumCardsInDrawPile(), gs.DrawPile.Len())
	}
	if gs.DrawPile.Len() != gs.Player1Info.NumCardsInDrawPile() {
		return fmt.Errorf("player %v thinks draw pile has %d cards, actually %d",
			Player1, gs.Player1Info.NumCardsInDrawPile(), gs.DrawPile.Len())
	}

	// All fixed draw pile cards must be in the draw pile.
	for i := 0; i < gs.DrawPile.Len(); i++ {
		card := gs.FixedDrawPileCards.NthCard(i)
		if card != cards.Unknown && gs.DrawPile.CountOf(card) == 0 {
			return fmt.Errorf("card %v fixed at position %v in draw pile but not in set %v",
				card, i, gs.DrawPile)
		}
	}

	// If a draw pile card is fixed in the view of either player,
	// it must be fixed here as well.
	for i := 0; i < gs.DrawPile.Len(); i++ {
		p0Card := gs.Player0Info.KnownDrawPileCards.NthCard(i)
		if p0Card != cards.Unknown && p0Card != gs.FixedDrawPileCards.NthCard(i) {
			return fmt.Errorf("player %v thinks draw pile position %d is %v, but actually %v",
				Player0, i, p0Card, gs.FixedDrawPileCards.NthCard(i))
		}

		p1Card := gs.Player1Info.KnownDrawPileCards.NthCard(i)
		if p1Card != cards.Unknown && p1Card != gs.FixedDrawPileCards.NthCard(i) {
			return fmt.Errorf("player %v thinks draw pile position %d is %v, but actually %v",
				Player1, i, p1Card, gs.FixedDrawPileCards.NthCard(i))
		}
	}

	return nil
}

func (gs *GameState) InfoSet(p Player) *InfoSet {
	if p == Player0 {
		return &gs.Player0Info
	}

	return &gs.Player1Info
}

func (gs *GameState) GetPlayerHand(p Player) cards.Set {
	return gs.InfoSet(p).OurHand
}

// GameNode represents a state of play in the extensive-form game tree.
type GameNode struct {
	state    GameState
	player   Player
	children []*GameNode
	// If player == Chance, then these are the cumulative probabilities
	// of selecting each of the children. i.e. to select a child outcome,
	// draw a random number p \in (0, 1] and then select the first child with
	// cumulative probability >= p: children[cumulativeProbs >= p][0]
	cumulativeProbs []float64
}

func (gn *GameNode) SampleOne() *GameNode {
	if gn.player != Chance {
		panic(fmt.Errorf("cannot sample game child from non-chance %v node", gn.player))
	}

	x := rand.Float64()
	for i, p := range gn.cumulativeProbs {
		if p >= x {
			return gn.children[i]
		}
	}

	return nil
}

func (gn *GameNode) NumChildren() int {
	return len(gn.children)
}

func NewGameTree() *GameNode {
	glog.Info("Building game tree")
	// TODO: should not be uniform distribution over possible initial deals,
	// some are more likely than others.
	initialStates := enumerateInitialStates()
	uniform := make([]float64, len(initialStates))
	for i := 0; i < len(uniform); i++ {
		uniform[i] = float64(i+1) / float64(len(uniform))
	}

	return &GameNode{
		player:          Chance,
		children:        initialStates,
		cumulativeProbs: uniform,
	}
}

func enumerateInitialStates() []*GameNode {
	result := make([]*GameNode, 0)
	// Deal 4 cards to player 0.
	player0Deals := enumerateInitialDeals(cards.CoreDeck, cards.Set{}, cards.Unknown, 4, nil)
	glog.V(1).Infof("Enumerated %d initial deals for Player0", len(player0Deals))
	for _, p0Deal := range player0Deals {
		remainingCards := cards.CoreDeck
		remainingCards.RemoveAll(p0Deal)
		// Deal 4 cards to player 1.
		player1Deals := enumerateInitialDeals(remainingCards, cards.Set{}, cards.Unknown, 4, nil)
		glog.V(1).Infof("Enumerated %d initial deals for Player1", len(player1Deals))
		for _, p1Deal := range player1Deals {
			glog.Infof("P0 deal: %v, P1 deal: %v", p0Deal, p1Deal)
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

	children, probs := buildDrawCardChildren(state, player, fromBottom, pendingTurns)
	return &GameNode{
		state:           state,
		player:          Chance,
		children:        children,
		cumulativeProbs: probs,
	}
}

func newPlayTurnNode(state GameState, player Player, pendingTurns int) *GameNode {
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
		state:    state,
		player:   player,
		children: buildPlayTurnChildren(state, player, pendingTurns),
	}
}

func newGiveCardNode(state GameState, player Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building give card node: player = %v, pending turns = %v",
		player, pendingTurns)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	return &GameNode{
		state:    state,
		player:   player,
		children: buildGiveCardChildren(state, player, pendingTurns),
	}
}

func newMustDefuseNode(state GameState, player Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building must defuse node: player = %v, pending turns = %v",
		player, pendingTurns)

	newState := state
	newState.InfoSet(player).PlayCard(cards.Defuse)
	newState.InfoSet(1 - player).OpponentPlayedCard(cards.Defuse)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	// Player may choose where to place exploding cat back in the draw pile.
	return &GameNode{
		state:    newState,
		player:   player,
		children: buildDefuseChildren(newState, player, pendingTurns),
	}
}

func newSeeTheFutureNode(state GameState, player Player, pendingTurns int) *GameNode {
	glog.V(3).Infof("Building see the future node: player = %v, pending turns = %v",
		player, pendingTurns)
	if err := state.Validate(); err != nil {
		panic(err)
	}

	children, probs := buildSeeTheFutureChildren(state, player, pendingTurns)
	return &GameNode{
		state:           state,
		player:          Chance,
		children:        children,
		cumulativeProbs: probs,
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
		card := state.FixedDrawPileCards.NthCard(bottom)
		nextNode := getNextDrawnCardNode(state, player, card, fromBottom, pendingTurns)
		result = append(result, nextNode)
		probs = append(probs, 1.0)
		return result, probs
	} else if !fromBottom && topCardIsKnown(state) {
		card := state.FixedDrawPileCards.NthCard(0)
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

	glog.V(2).Infof("Built %d draw card children with probs: %v", len(result), probs)
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
			glog.Info("Reached terminal node")
			nextNode = &GameNode{state: newState, player: 1 - player}
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
	// Shift known draw pile cards up by one.
	position := 0
	if fromBottom {
		position = newState.DrawPile.Len() - 1
	}
	newState.FixedDrawPileCards.RemoveCard(position)
	// Add to player's hand.
	newState.InfoSet(player).DrawCard(card, fromBottom)
	newState.InfoSet(1-player).OpponentDrewCard(card, fromBottom)
	return newState
}

func buildSeeTheFutureChildren(state GameState, player Player, pendingTurns int) ([]*GameNode, []float64) {
	var result []*GameNode
	var cumulativeProbs []float64

	cards, probs := enumerateTopNCards(state.DrawPile, state.FixedDrawPileCards, 3)
	var cumProb float64
	for i, topN := range cards {
		cumProb += probs[i]
		cumulativeProbs = append(cumulativeProbs, cumProb)

		newState := state
		newState.InfoSet(player).SeeTopCards(topN)
		for j, card := range topN {
			newState.FixedDrawPileCards.SetNthCard(j, card)
		}
		newNode := newPlayTurnNode(newState, player, pendingTurns)
		result = append(result, newNode)
	}

	glog.V(2).Infof("Built %d see the future children", len(result))
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
		if n == 1 {
			result = append(result, []cards.Card{card})
			resultProbs = append(resultProbs, p)
		} else { // Recurse to enumerate remaining n-1 cards.
			remainingDrawPile := drawPile
			remainingDrawPile[card]--
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
	return state.FixedDrawPileCards.NthCard(0) != cards.Unknown
}

func bottomCardIsKnown(state GameState) bool {
	bottom := state.DrawPile.Len() - 1
	return state.FixedDrawPileCards.NthCard(bottom) != cards.Unknown
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
		glog.Infof("Player %v playing %v", player, card)
		glog.Infof("Game state is: %+v", state)
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

	glog.V(2).Infof("Built %d play turn children", len(result))
	return result
}

func shuffleDrawPile(state GameState) GameState {
	result := state
	result.FixedDrawPileCards = cards.NewStack(nil)
	result.Player0Info.KnownDrawPileCards = cards.NewStack(nil)
	result.Player1Info.KnownDrawPileCards = cards.NewStack(nil)
	return result
}

func buildGiveCardChildren(state GameState, player Player, pendingTurns int) []*GameNode {
	cardChoices := state.GetPlayerHand(player).Distinct()
	result := make([]*GameNode, 0, len(cardChoices))
	for _, card := range cardChoices {
		glog.V(3).Infof("Player %v giving card %v", player, card)
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

	glog.V(2).Infof("Built %d give card children", len(result))
	return result
}

func buildDefuseChildren(state GameState, player Player, pendingTurns int) []*GameNode {
	result := make([]*GameNode, 0)
	nCardsInDrawPile := uint8(state.DrawPile.Len())
	nOptions := int(min(nCardsInDrawPile, 5))
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
	glog.V(2).Infof("Built %d defuse children", len(result))
	return result
}

func insertExplodingCat(state GameState, player Player, position int) GameState {
	newState := state
	// Place exploding cat card in the Nth position in draw pile.
	newState.DrawPile.Add(cards.ExplodingCat)
	newState.FixedDrawPileCards.InsertCard(cards.ExplodingCat, position)
	newState.InfoSet(player).OurHand.Remove(cards.ExplodingCat)
	newState.InfoSet(player).KnownDrawPileCards.InsertCard(cards.ExplodingCat, position)
	newState.InfoSet(1 - player).OpponentHand.Remove(cards.ExplodingCat)
	// FIXME: Known draw pile cards is not completely reset,
	// just reset until we get to the cat. Player's knowledge beneath the
	// insertion should be retained!
	newState.InfoSet(1 - player).KnownDrawPileCards = cards.NewStack(nil)
	return newState
}

func enumerateInitialDeals(available cards.Set, current cards.Set, start cards.Card, desired int, result []cards.Set) []cards.Set {
	nRemaining := uint8(desired - current.Len())
	if nRemaining == 0 {
		return append(result, current)
	}

	for card := start; card <= cards.Cat; card++ {
		count := available[card]
		for i := uint8(0); i <= min(count, nRemaining); i++ {
			current[card] += i
			available[card] -= i
			result = enumerateInitialDeals(available, current, card+1, desired, result)
			current[card] -= i
			available[card] += i
		}
	}

	return result
}

func min(i, j uint8) uint8 {
	if i < j {
		return i
	}

	return j
}
