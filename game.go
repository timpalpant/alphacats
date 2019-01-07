package alphacats

import (
	"github.com/timpalpant/alphacats/cards"
)

type Player uint8

const (
	Player0 Player = iota
	Player1
	Chance
)

func nextPlayer(p Player) Player {
	if p == Player0 {
		return Player1
	}

	return Player0
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

func NewGameTree() *GameNode {
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
	for _, p0Deal := range player0Deals {
		remainingCards := cards.CoreDeck
		remainingCards.RemoveAll(p0Deal)
		// Deal 4 cards to player 1.
		player1Deals := enumerateInitialDeals(remainingCards, cards.Set{}, cards.Unknown, 4, nil)
		for _, p1Deal := range player1Deals {
			gameState := buildInitialGameState(p0Deal, p1Deal)
			// Player0 always goes first.
			node := newPlayTurnNode(gameState, Player0, 1)
			result = append(result, node)
		}
	}

	return result
}

func buildInitialGameState(player1Deal, player2Deal cards.Set) GameState {
	remainingCards := cards.CoreDeck
	remainingCards.RemoveAll(player1Deal)
	remainingCards.RemoveAll(player2Deal)
	return GameState{
		DrawPile:    remainingCards,
		Player0Info: NewInfoSetFromInitialDeal(player1Deal),
		Player1Info: NewInfoSetFromInitialDeal(player2Deal),
	}
}

// Chance node where the given player is drawing a card from the draw pile.
// If fromBottom == true, then the player is drawing from the bottom of the pile.
func newDrawCardNode(state GameState, player Player, fromBottom bool, pendingTurns int) *GameNode {
	children, probs := buildDrawCardChildren(state, player, fromBottom, pendingTurns)
	return &GameNode{
		state:           state,
		player:          Chance,
		children:        children,
		cumulativeProbs: probs,
	}
}

func newPlayTurnNode(state GameState, player Player, pendingTurns int) *GameNode {
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
	return &GameNode{
		state:    state,
		player:   player,
		children: buildGiveCardChildren(state, player, pendingTurns),
	}
}

func newMustDefuseNode(state GameState, player Player, pendingTurns int) *GameNode {
	newState := state
	newState.InfoSet(player).PlayCard(cards.Defuse)
	newState.InfoSet(1 - player).OpponentPlayedCard(cards.Defuse)

	// Player may choose where to place exploding cat back in the draw pile.
	return &GameNode{
		state:    newState,
		player:   player,
		children: buildDefuseChildren(newState, player, pendingTurns),
	}
}

func playerDrewCard(state GameState, player Player, card cards.Card, fromBottom bool) GameState {
	newState := state
	// Pop card from the draw pile.
	newState.DrawPile[card]--
	// Shift known draw pile cards up by one.
	position := 0
	if fromBottom {
		position = newState.DrawPile.Len() - 1
	}
	newState.FixedDrawPileCards.RemoveCard(position)
	// Add to player's hand.
	newState.InfoSet(player).DrawCard(card, fromBottom)
	newState.InfoSet(1 - player).OpponentDrewCard(fromBottom)
	return newState
}

func buildDrawCardChildren(state GameState, player Player, fromBottom bool, pendingTurns int) ([]*GameNode, []float64) {
	var result []*GameNode
	var probs []float64
	pendingTurns--

	// Check if card being drawn is known. If it is then this node is
	// deterministic, return it.
	if fromBottom && bottomCardIsKnown(state) {
		bottom := state.DrawPile.Len() - 1
		drawnCard := state.FixedDrawPileCards.NthCard(bottom)
		newState := playerDrewCard(state, player, drawnCard, fromBottom)
		nextNode := newPlayTurnNode(newState, player, pendingTurns)
		result = append(result, nextNode)
		probs = append(probs, 1.0)
		return result, probs
	} else if !fromBottom && topCardIsKnown(state) {
		drawnCard := state.FixedDrawPileCards.NthCard(0)
		newState := playerDrewCard(state, player, drawnCard, fromBottom)
		nextNode := newPlayTurnNode(newState, player, pendingTurns)
		result = append(result, nextNode)
		probs = append(probs, 1.0)
		return result, probs
	}

	// Else: choose a card randomly according to distribution.
	cumProb := 0.0
	for _, card := range state.DrawPile.Distinct() {
		newState := playerDrewCard(state, player, card, fromBottom)
		p := float64(state.DrawPile.CountOf(card)) / float64(state.DrawPile.Len())
		cumProb += p

		var nextNode *GameNode
		if card == cards.ExplodingCat {
			// FIXME: Both players see that the card drawn was a cat,
			// even if they didn't know it already.

			if playerHasDefuseCard(state, player) {
				// Player has a defuse card, must play it.
				nextNode = newMustDefuseNode(newState, player, pendingTurns)
			} else {
				// Player does not have a defuse card, end game with loss for them.
				nextNode = &GameNode{player: player}
			}
		} else {
			// Just a normal card, add it to player's hand and continue.
			nextNode = newPlayTurnNode(newState, player, pendingTurns)
		}

		result = append(result, nextNode)
		probs = append(probs, cumProb)
	}

	return result, probs
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
		newGameState.InfoSet(player).PlayCard(card)
		newGameState.InfoSet(1 - player).OpponentPlayedCard(card)

		var nextNode *GameNode
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
		}

		result = append(result, nextNode)
	}

	// End our turn by drawing a card.
	nextNode := newDrawCardNode(state, player, false, pendingTurns)
	result = append(result, nextNode)

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
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Adding card to opponent's hand,
		//   3) Returning to opponent's turn.
		newGameState := state

		ourInfo := newGameState.InfoSet(player)
		ourInfo.OurHand[card]--
		ourInfo.OpponentHand[card]++

		opponentInfo := newGameState.InfoSet(1 - player)
		opponentInfo.OurHand[card]++
		if opponentInfo.OpponentHand[card] > 0 {
			// If opponent already knew we had one of these cards
			opponentInfo.OpponentHand[card]--
		} else {
			// Otherwise it was one of the Unknown cards in our hand.
			opponentInfo.OpponentHand[cards.Unknown]--
			opponentInfo.RemainingCards[card]--
		}

		// Game play returns to other player (with the given card in their hand).
		nextNode := newPlayTurnNode(newGameState, nextPlayer(player), pendingTurns)
		result = append(result, nextNode)
	}

	return result
}

func buildDefuseChildren(state GameState, player Player, pendingTurns int) []*GameNode {
	result := make([]*GameNode, 0)
	nCardsInDrawPile := state.DrawPile.Len()
	if nCardsInDrawPile < 5 {
		for i := 0; i <= nCardsInDrawPile; i++ {
			// Place exploding cat card in the Nth position in draw pile.
			state.DrawPile[cards.ExplodingCat]++
			state.FixedDrawPileCards.InsertCard(cards.ExplodingCat, i)
			state.InfoSet(player).DrawPile[cards.ExplodingCat]++
			state.InfoSet(player).KnownDrawPileCards.InsertCard(cards.ExplodingCat, i)
			state.InfoSet(1 - player).DrawPile[cards.ExplodingCat]++
			// FIXME: Known draw pile cards is not completely reset,
			// just reset until we get to the cat. Player's knowledge beneath the
			// insertion should be retained!
			state.InfoSet(1 - player).KnownDrawPileCards = cards.NewStack(nil)
		}
	} else {
		for i := 0; i <= 5; i++ {
			// Place exploding cat in the Nth position in draw pile.
		}

		// Place exploding cat on the bottom of the draw pile.
	}

	// TODO: Place randomly?
	return result
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
