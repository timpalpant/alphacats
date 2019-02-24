package gamestate

import (
	"crypto/md5"
	"encoding/binary"
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
	PositionInDrawPile uint8         // Private information.
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
		if a.CardsSeen[1] != cards.Unknown || a.CardsSeen[2] != cards.Unknown {
			s += fmt.Sprintf(":%v", a.CardsSeen)
		} else {
			s += fmt.Sprintf(":%v", a.CardsSeen[0])
		}
	}
	return s
}

const MaxNumActions = 48

// History records the history of game actions to reach this state.
// It is bit-packed and pre-sized to avoid allocations.
// The first byte is public information. The other two bytes are private
// information to the player that performed the action.
// History is presized, rather than a slice, to reduce allocations.
type History struct {
	actions [MaxNumActions][3]byte
	n       int
}

func (h *History) String() string {
	return fmt.Sprintf("%v", h.AsSlice())
}

func (h *History) Len() int {
	return h.n
}

func (h *History) Get(i int) Action {
	if i >= h.n {
		panic(fmt.Errorf("index out of range: %d %v", i, h))
	}

	return decodeAction(h.actions[i])
}

func (h *History) Append(action Action) {
	if h.n >= len(h.actions) {
		panic(fmt.Errorf("history exceeded max length: %v", h))
	}

	h.actions[h.n] = encodeAction(action)
	h.n++
}

func (h *History) AsSlice() []Action {
	result := make([]Action, h.n)
	for i, packed := range h.actions[:h.n] {
		result[i] = decodeAction(packed)
	}
	return result
}

// Gets the full, unabstracted infoset (but hashed into md5).
func (h *History) GetInfoSet(player Player, hand cards.Set, nCardsInDrawPile int) InfoSet {
	return InfoSet{
		History:            h.asViewedBy(player),
		Hand:               hand,
		NumCardsInDrawPile: nCardsInDrawPile,
	}
}

// Censor the given full game history to contain only info available
// to the given player.
func (h *History) asViewedBy(player Player) History {
	result := *h
	for i := 0; i < h.n; i++ {
		if Player(result.actions[i][0]&0x1) == player { // We don't want to fully decode.
			result.actions[i][1] = 0
			result.actions[i][2] = 0
		}
	}

	return result
}

// Action is packed as bits within a [3]uint8:
//   [0] Player (0 or 1)
//   [1-3] Type (1 - 4)
//   [4-7] Card (1 - 10)
//   [8-11] PositionInDrawPile (0 - 13)
//   [12-24] 3 Cards (1 - 10)
// Thus the first byte is public info, the second two bytes are private info.
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
		PositionInDrawPile: uint8(packed[1] & 0xf),
		CardsSeen: [3]cards.Card{
			cards.Card(packed[1] >> 4),
			cards.Card(packed[2] & 0xf),
			cards.Card(packed[2] >> 4),
		},
	}
}

type InfoSet struct {
	History            History
	Hand               cards.Set
	NumCardsInDrawPile int
}

func (is InfoSet) Key() string {
	var buf [3*MaxNumActions + 8]byte
	for i := 0; i < is.History.Len(); i++ {
		packed := is.History.actions[i]
		buf[i] = packed[0]
		buf[i+1] = packed[1]
		buf[i+2] = packed[2]
	}

	// Player's hand is appended to private game history.
	binary.LittleEndian.PutUint64(buf[3*is.History.Len():], uint64(is.Hand))

	// Hash into smaller bitstring since it is sparse.
	// We'll hope for no collisions :)
	hash := md5.Sum(buf[:3*is.History.Len()+8])
	return string(hash[:])
}
