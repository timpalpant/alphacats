package gamestate

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
)

// GameState represents the current state of the game.
type GameState struct {
	// The history of player actions that were taken to reach this state.
	history History
	// Set of Cards remaining in the draw pile.
	drawPile    cards.Stack
	player0Hand cards.Set
	player1Hand cards.Set

	// These are lazily constructed by calls to GetInfoSet,
	// but we want to preallocate the memory for them to reduce garbage.
	player0InfoSet InfoSet
	player1InfoSet InfoSet
}

// New returns a new GameState created with the given draw pile and deals
// of cards to each of the players.
func New(drawPile cards.Stack, player0Deal, player1Deal cards.Set) GameState {
	return GameState{
		drawPile:    drawPile,
		player0Hand: player0Deal,
		player1Hand: player1Deal,
	}
}

// NewShuffled returns a new GameState created by applying the given shuffling
// of the draw pile to an existing GameState.
func NewShuffled(prevState GameState, newDrawPile cards.Stack) GameState {
	result := prevState
	result.drawPile = newDrawPile
	return result
}

// Apply returns the new GameState created by applying the given Action.
func (gs *GameState) Apply(action Action) {
	switch action.Type {
	case PlayCard:
		action = gs.playCard(action)
	case DrawCard:
		action = gs.drawCard(action)
	case GiveCard:
		gs.giveCard(action.Player, action.Card)
	case InsertExplodingCat:
		gs.insertExplodingCat(action.Player, int(action.PositionInDrawPile))
	default:
		panic(fmt.Errorf("invalid action: %+v", action))
	}

	gs.history.Append(action)
}

func (gs *GameState) insertExplodingCat(player Player, position int) {
	if player == Player0 {
		gs.player0Hand.Remove(cards.ExplodingCat)
	} else {
		gs.player1Hand.Remove(cards.ExplodingCat)
	}
	gs.drawPile.InsertCard(cards.ExplodingCat, position)
}

func (gs *GameState) String() string {
	return fmt.Sprintf("draw pile: %s, p0: %s, p1: %s. history: %s",
		gs.drawPile, gs.player0Hand, gs.player1Hand, gs.history.String())
}

func (gs *GameState) GetDrawPile() cards.Stack {
	return gs.drawPile
}

func (gs *GameState) GetPlayerHand(p Player) cards.Set {
	if p == Player0 {
		return gs.player0Hand
	}

	return gs.player1Hand
}

func (gs *GameState) LastAction() Action {
	if gs.history.Len() == 0 {
		return Action{}
	}

	return gs.history.Get(gs.history.Len() - 1)
}

// InfoSet represents the state of the game from the point of view of one of the
// players. Note that multiple distinct game states may have the same InfoSet
// due to hidden information that the player is not privy to.
func (gs *GameState) GetInfoSet(player Player) *InfoSet {
	hand := gs.player0Hand
	if player == Player1 {
		hand = gs.player1Hand
	}

	// We store the InfoSet on the GameState and return a reference to it,
	// rather than returning the *InfoSet directly, to reduce garbage
	// (the InfoSets are allocated inline with the GameState struct).
	// We can't just return an InfoSet value type because it will have to
	// be copied to the heap (convT2Inoptr) to satisfy the cfr.InfoSet interface.
	// https://science.raphael.poss.name/go-calling-convention-x86-64.html#id25
	// This was measured to be ~10% faster.
	is := gs.history.GetInfoSet(player, hand)
	if player == Player0 {
		gs.player0InfoSet = is
		return &gs.player0InfoSet
	} else {
		gs.player1InfoSet = is
		return &gs.player1InfoSet
	}
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

func (gs *GameState) playCard(action Action) Action {
	if action.Player == Player0 {
		gs.player0Hand.Remove(action.Card)
	} else {
		gs.player1Hand.Remove(action.Card)
	}

	switch action.Card {
	case cards.SeeTheFuture:
		action.CardsSeen = [3]cards.Card{
			gs.drawPile.NthCard(0),
			gs.drawPile.NthCard(1),
			gs.drawPile.NthCard(2),
		}
	case cards.DrawFromTheBottom:
		drawn := gs.drawPile.NthCard(gs.drawPile.Len() - 1)
		action.CardsSeen[0] = drawn
		gs.drawPile.RemoveCard(gs.drawPile.Len() - 1)
		if action.Player == Player0 {
			gs.player0Hand.Add(drawn)
		} else {
			gs.player1Hand.Add(drawn)
		}
	}

	return action
}

func (gs *GameState) drawCard(action Action) Action {
	drawn := gs.drawPile.NthCard(0)
	gs.drawPile.RemoveCard(0)
	if action.Player == Player0 {
		gs.player0Hand.Add(drawn)
	} else {
		gs.player1Hand.Add(drawn)
	}
	action.CardsSeen[0] = drawn
	return action
}
