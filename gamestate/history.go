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
)

var allActions = []ActionType{
	DrawCard,
	PlayCard,
	GiveCard,
	InsertExplodingCat,
}

var actionTypeStr = [...]string{
	"Invalid",
	"DrawCard",
	"PlayCard",
	"GiveCard",
	"InsertExplodingCat",
}

func (t ActionType) String() string {
	return actionTypeStr[t]
}

// Action records each transition/edge in the game history.
type Action struct {
	Player             Player
	Type               ActionType
	Card               cards.Card
	PositionInDrawPile int           // Private information.
	CardsSeen          [3]cards.Card // Private information.
}

func (a Action) String() string {
	s := fmt.Sprintf("%s:%s", a.Player, a.Type)
	if a.Card != cards.Unknown {
		s += ":" + a.Card.String()
	}
	if a.Type == InsertExplodingCat {
		s += fmt.Sprintf(":%d", a.PositionInDrawPile)
	}
	if a.CardsSeen[0] != cards.Unknown {
		s += fmt.Sprintf(":%s", a.CardsSeen)
	}
	return s
}

const MaxNumActions = 48

// History records the history of game actions to reach this state.
// It is bit-packed and pre-sized to avoid allocations.
// The first byte is public information. The other two bytes are private
// information to the player that performed the action.
// History is presized, rather than a slice, to reduce allocations.
type history struct {
	actions [MaxNumActions][3]uint8
	n       int
}

func (h *history) String() string {
	return fmt.Sprintf("%v", h.AsSlice())
}

func (h *history) Len() int {
	return h.n
}

func (h *history) Get(i int) Action {
	if i >= h.n {
		panic(fmt.Errorf("index out of range: %d %v", i, h))
	}

	return decodeAction(h.actions[i])
}

func (h *history) Append(action Action) {
	if h.n >= len(h.actions) {
		panic(fmt.Errorf("history exceeded max length: %v", h))
	}

	h.actions[h.n] = encodeAction(action)
	h.n++
}

func (h *history) AsSlice() []Action {
	result := make([]Action, h.n)
	for i, packed := range h.actions[:h.n] {
		result[i] = decodeAction(packed)
	}
	return result
}

// InfoSet is encoded with public history first, so that we
// can compress all private states with the same public prefix.
func (h *history) EncodeInfoSet(player Player, buf []byte) int {
	for i, packed := range h.actions[:h.n] {
		packed = censorAction(packed, player)
		copy(buf[3*i:], packed[:])
	}

	return 3 * h.n
}

// Remove Action info that is not privy to the given player.
func censorAction(packed [3]uint8, p Player) [3]uint8 {
	player := Player(packed[0] & 0x1)
	if player == p { // Action was by this player, return unchanged.
		return packed
	}

	actionType := ActionType((packed[0] >> 1) & 0x7)
	if actionType == PlayCard || actionType == DrawCard || actionType == InsertExplodingCat {
		// Just keep first byte (public info: Player + Type + Card played)
		packed[1] = 0
		packed[2] = 0
	}

	return packed
}

// Action is packed as bits within a [3]uint8:
// [0] Player
// [1-3] Type
// [4-7] Card
// [8-11] PositionInDrawPile (0 - 13)
// [12-24] 3 Cards
func encodeAction(a Action) [3]uint8 {
	var result [3]uint8
	result[0] = uint8(a.Player)
	result[0] += uint8(a.Type << 1)
	result[0] += uint8(a.Card << 4)
	result[1] = uint8(a.PositionInDrawPile)
	result[1] += uint8(a.CardsSeen[0] << 4)
	result[2] = uint8(a.CardsSeen[1])
	result[2] += uint8(a.CardsSeen[2] << 4)
	return result
}

func decodeAction(packed [3]uint8) Action {
	return Action{
		Player:             Player(packed[0] & 0x1),
		Type:               ActionType((packed[0] >> 1) & 0x7),
		Card:               cards.Card(packed[0] >> 4),
		PositionInDrawPile: int(packed[1] & 0xf),
		CardsSeen: [3]cards.Card{
			cards.Card(packed[1] >> 4),
			cards.Card(packed[2] & 0xf),
			cards.Card(packed[2] >> 4),
		},
	}
}
