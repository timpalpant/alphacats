package gamestate

import (
	"fmt"

	"github.com/timpalpant/alphacats/cards"
)

// ActionType is the type of action a player performed in the game's history.
type ActionType uint8

const (
	_ ActionType = iota
	DrawCard
	PlayCard
	GiveCard
	InsertExplodingCat
	SeeTheFuture
)

var actionTypeStr = [...]string{
	"Invalid",
	"DrawCard",
	"PlayCard",
	"GiveCard",
	"InsertExplodingCat",
	"SeeTheFuture",
}

func (t ActionType) String() string {
	return actionTypeStr[t]
}

// Action records each player choice in the game history.
type Action struct {
	Player             Player
	Type               ActionType
	Card               cards.Card   // May be private information.
	PositionInDrawPile int          // May be private information.
	Cards              []cards.Card // May be private information.
}

func (a Action) String() string {
	return fmt.Sprintf("%s:%s:%s", a.Player, a.Type, a.Card)
}

// History is a bit-packed representation of a slice of Actions
// representing the public history of the game.
//
// Because we know the maximum number of turns in a game, we can
// pre-allocate an appropriately sized array within the GameState
// struct, reducing allocations required for the game history.
type publicHistory [16]uint8

func newHistoryFromSlice(actions []Action) publicHistory {
	history := publicHistory{}
	for _, action := range actions {
		history.Append(action)
	}
	return history
}

func (h publicHistory) AsSlice() []Action {
	var result []Action
	for i := 0; i < len(h); i++ {
		if h[i] == 0 {
			break
		}

		action := decodeAction(h[i])
		result = append(result, action)
	}

	return result
}

func (h *publicHistory) Append(a Action) {
	if h[len(h)-1] != 0 {
		panic("overflow in public history")
	}

	// Shift all elements right one.
	for i := len(h) - 1; i > 0; i++ {
		h[i] = h[i-1]
	}

	// Insert new element at the front.
	h[0] = encodeAction(a)
}

func encodeAction(a Action) uint8 {
	// [0]: Lowest-order bit is 0/1 for Player.
	// [1-3]: Next 3 bits are the Type.
	// [4-7]: Next 4 bits are the Card.
	// PositionInDrawPile and Cards are not encoded,
	// since they are private information to one of the players.
	result := uint8(a.Player)
	result += uint8(a.Type) << 1
	result += uint8(a.Card) << 4
	return result
}

func decodeAction(bits uint8) Action {
	return Action{
		Player: Player(uint8(bits & 0x1)),     // 0b00000001
		Type:   ActionType(uint8(bits & 0x6)), // 0b00000110
		Card:   cards.Card(int(bits & 0xf0)),  // 0b11110000
	}
}
