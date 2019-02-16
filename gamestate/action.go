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

var allActions = []ActionType{
	DrawCard,
	PlayCard,
	GiveCard,
	InsertExplodingCat,
	SeeTheFuture,
}

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
	s := fmt.Sprintf("%s:%s", a.Player, a.Type)
	if a.Card != cards.Unknown {
		s += ":" + a.Card.String()
	}
	if a.PositionInDrawPile != 0 {
		s += fmt.Sprintf(":%d", a.PositionInDrawPile)
	}
	if len(a.Cards) != 0 {
		s += fmt.Sprintf(":%v", a.Cards)
	}
	return s
}

// History is a bit-packed representation of a slice of Actions
// representing the public history of the game.
//
// Because we know the maximum number of turns in a game, we can
// pre-allocate an appropriately sized array within the GameState
// struct, reducing allocations required for the game history.
type publicHistory struct {
	packed [48]uint8
	n      uint8
}

func newHistoryFromSlice(actions []Action) publicHistory {
	history := publicHistory{}
	for _, action := range actions {
		history.Append(action)
	}
	return history
}

func (h publicHistory) AsSlice() []Action {
	var result []Action
	for i := uint8(0); i < h.n; i++ {
		action := decodeAction(h.packed[i])
		result = append(result, action)
	}

	return result
}

func (h *publicHistory) Len() int {
	return int(h.n)
}

func (h *publicHistory) Append(a Action) {
	if int(h.n) >= len(h.packed) {
		panic("overflow in public history")
	}

	h.packed[h.n] = encodeAction(a)
	h.n++
}

func (h *publicHistory) Get(i int) Action {
	if i >= int(h.n) {
		panic(fmt.Errorf("%d out of range for history with %d elements", i, h.n))
	}

	return decodeAction(h.packed[i])
}

func encodeAction(a Action) uint8 {
	// [0]: Lowest-order bit is 0/1 for Player.
	// [1-3]: Next 3 bits are the Type.
	// [4-7]: Next 4 bits are the Card.
	// PositionInDrawPile and Cards are not encoded,
	// since they are private information to one of the players.
	// Card is only encoded if the Type is PlayCard (public).
	result := uint8(a.Player)
	result += uint8(a.Type) << 1
	if a.Type == PlayCard {
		result += uint8(a.Card) << 4
	}
	return result
}

func decodeAction(bits uint8) Action {
	return Action{
		Player: Player(uint8(bits & 0x1)),        // 0b00000001
		Type:   ActionType(uint8(bits&0xe) >> 1), // 0b00001110
		Card:   cards.Card(int(bits&0xf0) >> 4),  // 0b11110000
	}
}
