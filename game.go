package alphacats

type Player uint8

const (
	Player0 Player = iota
	Player1
)

func nextPlayer(p Player) Player {
	if p == Player0 {
		return Player1
	}

	return Player0
}

// GameState represents the current state of the game.
type GameState struct {
	DrawPile CardSet
	// Cards in the draw pile whose identity has been fixed
	// because one of the players knows it (either because of a
	// SeeTheFuture card or replacing a card in the deck).
	FixedDrawPileCards CardPile
	InfoSets           map[Player]InfoSet
}

type GameNode struct {
	state           GameState
	player          Player
	children        []*GameNode
	cumulativeProbs []float64
}

type GameTree struct {
	initialStates []*GameNode
	strategies    map[Player]Strategy
}

func NewGameTree() *GameTree {
	return &GameTree{
		initialStates: enumerateInitialStates(),
	}
}

func enumerateInitialStates() []*GameNode {
	result := make([]*GameNode, 0)
	player1Deals := enumerateInitialDeals(CoreDeck, CardSet{}, Unknown, 4, nil)
	for _, p1Deal := range player1Deals {
		remainingCards := CoreDeck.Remove(p1Deal)
		player2Deals := enumerateInitialDeals(remainingCards, CardSet{}, Unknown, 4, nil)
		for _, p2Deal := range player2Deals {
			gameState := buildInitialGameState(p1Deal, p2Deal)
			// Player0 always goes first.
			node := newGameNode(gameState, PlayTurn, Player0)
			result = append(result, node)
		}
	}

	return result
}

func buildInitialGameState(player1Deal, player2Deal CardSet) GameState {
	remainingCards := CoreDeck.Remove(player1Deal).Remove(player2Deal)
	return GameState{
		DrawPile: remainingCards,
		InfoSets: map[Player]InfoSet{
			Player0: NewInfoSetFromInitialDeal(player1Deal),
			Player1: NewInfoSetFromInitialDeal(player2Deal),
		},
	}
}

func newDrawCardNode(state GameState, player Player, pendingTurns int) *GameNode {
	return &GameNode{
		state:           state,
		player:          player,
		children:        buildDrawCardChildren(state, player, pendingTurns),
		cumulativeProbs: probs,
	}
}

func newPlayTurnNode(state GameState, player Player, pendingTurns int) *GameNode {
	if pendingTurns == 0 {
		// Player's turn is done, next player.
		player = nextPlayer(player)
		pendingTurns = 1
	}

	children, probs := buildDrawCardChildren(state, player, pendingTurns)
	return &GameNode{
		state:           state,
		player:          player,
		children:        children,
		cumulativeProbs: probs,
	}
}

func newGiveCardNode(state GameState, player Player, pendingTurns int) *GameNode {
	return &GameNode{
		state:           state,
		player:          player,
		children:        buildGiveCardChildren(state, player, pendingTurns),
		cumulativeProbs: probs,
	}

}

func newMustDefuseNode(state GameState, player Player, pendingTurns int) *GameNode {
	newState := state
	newState.InfoSets[player] = newState.InfoSets[player].PlayCard(Defuse)
	newState.InfoSets[1-player] = newState.InfoSets[1-player].OpponentPlayedCard(Defuse)

	// Player may choose where to place exploding cat back in the draw pile.
	return &GameNode{
		state:    newState,
		player:   player,
		children: buildDefuseChildren(newState, player, pendingTurns),
	}
}

func topCardIsKnown(state GameState) bool {
	return state.FixedDrawPileCards.NthCard(0) != Unknown
}

func playerHasDefuseCard(state GameState, player Player) bool {
	return state.InfoSets[player].OurHand.CountOf(Defuse) > 0
}

func playerDrewCard(state GameState, card Card) GameState {
	newState := state
	// Pop card from the draw pile.
	newState.DrawPile[card]--
	// Shift known draw pile cards up by one.
	newState.FixedDrawPileCards = newState.FixedDrawPileCards.RemoveCard(0)
	// Add to player's hand.
	newState.InfoSets[player] = newState.InfoSets[player].DrawCard(card)
	newState.InfoSets[1-player] = newState.InfoSets[1-player].OpponentDrewCard()
	return newState
}

func buildDrawCardChildren(state GameState, player Player, pendingTurns int) ([]*GameNode, []float64) {
	var result []*GameNode
	var probs []float64
	pendingTurns--

	// Check if top card is known. If it is then this node is
	// deterministic, return it.
	if topCardIsKnown(state) {
		drawnCard := state.FixedDrawPileCards.NthCard(0)
		newState := playerDrewCard(state, drawnCard)
		nextNode := newPlayTurnNode(newState, player, pendingTurns)
		result = append(result, nextNode)
		probs = append(probs, 1.0)
		return result, probs
	}

	// Else: choose a card randomly according to distribution.
	cumProb := 0.0
	for _, card := range state.DrawPile.Distinct() {
		newState := playerDrewCard(state, card)
		p := float64(state.DrawPile.CountOf(card)) / float64(state.DrawPile.Len())
		cumProb += p

		var nextNode *GameNode
		if card == ExplodingCat {
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
}

func buildPlayTurnChildren(state GameState, player Player, pendingTurns int) []*GameNode {
	// Choose whether to play a card or draw.
	cardChoices := state.InfoSets[player].OurHand.Distinct()
	result := make([]*GameNode, 0, len(cardChoices)+1)

	// Play one of the cards in our hand.
	for _, card := range cardChoices {
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Updating opponent's view of the world to reflect played card,
		//   3) Updating game state based on action in card.
		newGameState := state

		nextNode := newPlayTurnNode(newGameState, player, pendingTurns)
		result = append(result, nextNode)
	}

	// End our turn by drawing a card.
	nextNode := newDrawCardNode(state, player, pendingTurns)
	result = append(result, nextNode)

	return result
}

func buildGiveCardChildren(state GameState, player Player, pendingTurns int) []*GameNode {
	cardChoices := state.InfoSets[player].OurHand.Distinct()
	result := make([]*GameNode, 0, len(cardChoices))
	for _, card := range cardChoices {
		// Form child node by:
		//   1) Removing card from our hand,
		//   2) Adding card to opponent's hand,
		//   3) Returning to opponent's turn.
		newGameState := state

		ourInfo := newGameState.InfoSets[player]
		ourInfo.OurHand[card]--
		ourInfo.OpponentHand[card]++
		newGameState.InfoSets[player] = ourInfo

		opponentInfo := newGameState.InfoSets[1-player]
		opponentInfo.OurHand[card]++
		if opponentInfo.OpponentHand[card] > 0 {
			// If opponent already knew we had one of these cards
			opponentInfo.OpponentHand[card]--
		} else {
			// Otherwise it was one of the Unknown cards in our hand.
			opponentInfo.OpponentHand[Unknown]--
			opponentInfo.RemainingCards[card]--
		}
		newGameState.InfoSets[1-player] = opponentInfo

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
			state.DrawPile[ExplodingCat]++
			state.FixedDrawPileCard = state.FixedDrawPileCards.InsertCard(ExplodingCat, i)
			state.InfoSets[player].DrawPile[ExplodingCat]++
			state.InfoSets[player].KnownDrawPileCards = state.InfoSets[player].KnownDrawPileCards.InsertCard(ExplodingCat, i)
			state.InfoSets[1-player].DrawPile[ExplodingCat]++
			// FIXME: Known draw pile cards is not completely reset,
			// just reset until we get to the cat. Our knowledge beneath the
			// insertion should be retained!
			state.InfoSets[1-player].KnownDrawPileCards = CardPile(0)
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

func enumerateInitialDeals(available CardSet, current CardSet, start Card, desired int, result []CardSet) []CardSet {
	nRemaining := uint8(desired - current.Len())
	if nRemaining == 0 {
		return append(result, current)
	}

	for card := start; card <= Cat; card++ {
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
