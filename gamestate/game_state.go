package gamestate

import (
	"encoding/binary"
	"fmt"

	"github.com/timpalpant/alphacats/cards"
)

// GameState represents the current state of the game.
//
// Any additional fields added to GameState must also be added to clone().
type GameState struct {
	// The history of player actions that were taken to reach this state.
	history history
	// Set of Cards remaining in the draw pile.
	// Note that the players will not in general have access to this information.
	drawPile    DrawPile
	player0Hand cards.Set
	player1Hand cards.Set
}

// New returns a new GameState created by dealing the given sets of cards
// to each player at the beginning of the game.
func New(player0Deal, player1Deal cards.Set) GameState {
	remainingCards := cards.CoreDeck
	remainingCards.RemoveAll(player0Deal)
	remainingCards.RemoveAll(player1Deal)
	remainingCards.Add(cards.ExplodingCat)
	player0Deal.Add(cards.Defuse)
	player1Deal.Add(cards.Defuse)
	return GameState{
		drawPile:    NewDrawPile(remainingCards),
		player0Hand: player0Deal,
		player1Hand: player1Deal,
	}
}

// Apply returns the new GameState created by applying the given Action.
func (gs *GameState) Apply(action Action) {
	switch action.Type {
	case PlayCard:
		gs.playCard(action.Player, action.Card)
	case DrawCard:
		gs.drawPile.drawCard(action.Card)
	case DrawCardFromBottom:
		gs.drawPile.drawCardFromBottom(action.Card)
	case GiveCard:
		gs.giveCard(action.Player, action.Card)
	case InsertExplodingCat:
		gs.drawPile.insert(cards.ExplodingCat, action.PositionInDrawPile)
	case SeeTheFuture:
		gs.drawPile.fixTop3Cards(action.Cards)
	default:
		panic(fmt.Errorf("invalid action: %+v", action))
	}

	gs.history.Append(action)
}

func (gs *GameState) String() string {
	return fmt.Sprintf("draw pile: %s, p0: %s, p1: %s",
		gs.drawPile, gs.player0Hand, gs.player1Hand)
}

func (gs *GameState) GetHistory() []Action {
	return gs.history.AsSlice()
}

func (gs *GameState) GetDrawPile() DrawPile {
	return gs.drawPile
}

func (gs *GameState) GetPlayerHand(p Player) cards.Set {
	if p == Player0 {
		return gs.player0Hand
	}

	return gs.player1Hand
}

func (gs *GameState) HasDefuseCard(p Player) bool {
	return gs.GetPlayerHand(p).Contains(cards.Defuse)
}

func (gs *GameState) LastActionWasSlap() bool {
	lastAction := gs.history.Get(gs.history.Len() - 1)
	return lastAction.Type == PlayCard && (lastAction.Card == cards.Slap1x || lastAction.Card == cards.Slap2x)
}

// InfoSet represents the state of the game from the point of view of one of the
// players. Note that multiple distinct game states may have the same InfoSet
// due to hidden information that the player is not privy to.
func (gs *GameState) GetInfoSet(player Player) string {
	nBytes := 8 + 4*gs.history.Len()
	buf := make([]byte, nBytes)
	if player == Player0 {
		binary.LittleEndian.PutUint64(buf, uint64(gs.player0Hand))
	} else {
		binary.LittleEndian.PutUint64(buf, uint64(gs.player1Hand))
	}

	gs.history.EncodeInfoSet(player, buf[8:])
	return string(buf)
}

func (gs *GameState) giveCard(player Player, card cards.Card) {
	if player == Player0 {
		gs.player0Hand.Remove(card)
		gs.player1Hand.Add(card)
	} else {
		gs.player1Hand.Remove(card)
		gs.player0Hand.Add(card)
	}
}

func (gs *GameState) playCard(player Player, card cards.Card) {
	if player == Player0 {
		gs.player0Hand.Remove(card)
	} else {
		gs.player1Hand.Remove(card)
	}
}
