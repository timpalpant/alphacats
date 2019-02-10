package gamestate

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"

	"github.com/timpalpant/alphacats/cards"
)

// GameState represents the current state of the game.
type GameState struct {
	// The history of player actions that were taken to reach this state.
	history []Action
	// Set of Cards remaining in the draw pile.
	// Note that the players will not in general have access to this information.
	drawPile cards.Set
	// The Cards that have been played in the discard pile.
	discardPile cards.Stack
	// Private information held from the point of view of either player.
	player0Info privateInfo
	player1Info privateInfo
}

// New returns a new GameState created by dealing the given sets of cards
// to each player at the beginning of the game.
func New(player0Deal, player1Deal cards.Set) GameState {
	remainingCards := cards.CoreDeck
	remainingCards.RemoveAll(player0Deal)
	remainingCards.RemoveAll(player1Deal)
	remainingCards.Add(cards.ExplodingCat)
	return GameState{
		drawPile:    remainingCards,
		player0Info: newPrivateInfo(player0Deal),
		player1Info: newPrivateInfo(player1Deal),
	}
}

// Apply returns the new GameState created by applying the given Action.
func Apply(state GameState, action Action) GameState {
	result := clone(state)
	result.history = append(result.history, action)

	switch action.Type {
	case DrawCard:
		result.drawCard(action.Player, action.Card, action.PositionInDrawPile)
	case PlayCard:
		result.playCard(action.Player, action.Card)
	case GiveCard:
		result.giveCard(action.Player, action.Card)
	case InsertExplodingCat:
		result.insertExplodingCat(action.Player, action.PositionInDrawPile)
	case SeeTheFuture:
		result.seeTopNCards(action.Player, action.Cards)
	default:
		panic(fmt.Errorf("invalid action: %+v", action))
	}

	return result
}

func clone(gs GameState) GameState {
	// Allocate with 1 extra capacity because we will always
	// be appending a new history item to the cloned state.
	historyCopy := make([]Action, len(gs.history)+1)
	copy(historyCopy, gs.history)

	return GameState{
		history:     historyCopy,
		drawPile:    gs.drawPile,
		discardPile: gs.discardPile,
		player0Info: gs.player0Info,
		player1Info: gs.player1Info,
	}
}

// Validate sanity checks the GameState to ensure we have maintained
// internal consistency in the game tree.
func (gs *GameState) Validate() error {
	if err := gs.player0Info.validate(); err != nil {
		return errors.Wrapf(err, "player %v info invalid", Player0)
	}

	if err := gs.player1Info.validate(); err != nil {
		return errors.Wrapf(err, "player %v info invalid", Player1)
	}

	// All fixed draw pile cards must be in the draw pile.
	for i := 0; i < gs.drawPile.Len(); i++ {
		card := gs.FixedDrawPileCards().NthCard(i)
		if card != cards.Unknown && gs.drawPile.CountOf(card) == 0 {
			return fmt.Errorf("card %v fixed at position %v in draw pile but not in set %v",
				card, i, gs.drawPile)
		}
	}

	// If a player knows a card in the other player's hand, they must
	// actually have it. Note: They may have more (that are Unknown top opponent).
	p0Unknown := gs.player0Info.opponentHand.CountOf(cards.Unknown)
	p1Unknown := gs.player1Info.opponentHand.CountOf(cards.Unknown)
	for card := cards.Unknown + 1; card <= cards.Cat; card++ {
		n := gs.player0Info.opponentHand.CountOf(card)
		m := gs.player1Info.ourHand.CountOf(card)
		if m < n || m > n+p0Unknown {
			return fmt.Errorf("player 0 thinks player 1 has %d of %v, but they actually have %d",
				n, card, m)
		}

		n = gs.player1Info.opponentHand.CountOf(card)
		m = gs.player0Info.ourHand.CountOf(card)
		if m < n || m > n+p1Unknown {
			return fmt.Errorf("player 1 thinks player 0 has %d of %v, but they actually have %d",
				n, card, m)
		}
	}

	return nil
}

func (gs *GameState) GetHistory() []Action {
	return gs.history
}

func (gs *GameState) GetDrawPile() cards.Set {
	return gs.drawPile
}

// FixedDrawPileCards is the union of all cards in the draw pile whose identity
// is fixed because either one of the players knows it.
func (gs *GameState) FixedDrawPileCards() cards.Stack {
	result := gs.player0Info.knownDrawPileCards
	for i := 0; i < gs.drawPile.Len(); i++ {
		card := gs.player1Info.knownDrawPileCards.NthCard(i)
		if card != cards.Unknown {
			p0View := result.NthCard(i)
			if p0View != cards.Unknown && p0View != card {
				panic(fmt.Errorf("inconsistent fixed card info: p0 = %v, p1 = %v",
					p0View, card))
			}

			result.SetNthCard(i, card)
		}
	}

	return result
}

func (gs *GameState) GetPlayerHand(p Player) cards.Set {
	return gs.privateInfo(p).ourHand
}

func (gs *GameState) HasDefuseCard(p Player) bool {
	return gs.GetPlayerHand(p).CountOf(cards.Defuse) > 0
}

// InfoSet represents the state of the game from the point of view of one of the
// players. Note that multiple distinct game states may have the same InfoSet
// due to hidden information that the player is not privy to.
type InfoSet struct {
	privateInfo privateInfo
	history     string
}

func (gs *GameState) GetInfoSet(player Player) InfoSet {
	return InfoSet{
		privateInfo: *gs.privateInfo(player),
		history:     historyString(gs.history),
	}
}

func historyString(actions []Action) string {
	parts := make([]string, len(actions))
	for i, action := range actions {
		parts[i] = action.String()
	}

	return strings.Join(parts, ",")
}

func (gs *GameState) privateInfo(p Player) *privateInfo {
	if p == Player0 {
		return &gs.player0Info
	}

	return &gs.player1Info
}

func (gs *GameState) playCard(player Player, card cards.Card) {
	gs.privateInfo(player).playCard(card)
	gs.privateInfo(1 - player).opponentPlayedCard(card)
	gs.discardPile.InsertCard(card, 0)
}

func (gs *GameState) drawCard(player Player, card cards.Card, position int) {
	// Pop card from the draw pile.
	gs.drawPile.Remove(card)
	// Add to player's hand.
	gs.privateInfo(player).drawCard(card, position)
	gs.privateInfo(1-player).opponentDrewCard(card, position)
}

func (gs *GameState) insertExplodingCat(player Player, position int) {
	// Place exploding cat card in the Nth position in draw pile.
	gs.drawPile.Add(cards.ExplodingCat)
	gs.privateInfo(player).ourHand.Remove(cards.ExplodingCat)
	gs.privateInfo(player).knownDrawPileCards.InsertCard(cards.ExplodingCat, position)
	gs.privateInfo(1 - player).opponentHand.Remove(cards.ExplodingCat)
	// FIXME: Known draw pile cards is not completely reset,
	// just reset until we get to the cat. Player's knowledge beneath the
	// insertion should be retained!
	gs.privateInfo(1 - player).knownDrawPileCards = cards.NewStack()
}

func (gs *GameState) shuffleDrawPile() {
	gs.player0Info.knownDrawPileCards = cards.NewStack()
	gs.player1Info.knownDrawPileCards = cards.NewStack()
}

func (gs *GameState) seeTopNCards(player Player, topN []cards.Card) {
	gs.privateInfo(player).seeTopCards(topN)
}

func (gs *GameState) giveCard(player Player, card cards.Card) {
	pInfo := gs.privateInfo(player)
	pInfo.ourHand.Remove(card)
	pInfo.opponentHand.Add(card)

	opponentInfo := gs.privateInfo(1 - player)
	opponentInfo.ourHand.Add(card)
	if opponentInfo.opponentHand.CountOf(card) > 0 {
		// If opponent already knew we had one of these cards
		opponentInfo.opponentHand.Remove(card)
	} else {
		// Otherwise it was one of the Unknown cards in our hand.
		opponentInfo.opponentHand.Remove(cards.Unknown)
	}
}
