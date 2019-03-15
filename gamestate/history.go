package gamestate

import (
	"crypto/md5"
	"encoding/binary"
	"encoding/gob"
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
// It is pre-sized to avoid allocations and keep GameState easily copyable.
type History struct {
	actions [MaxNumActions]Action
	n       int
}

func (h *History) String() string {
	return fmt.Sprintf("%v", h.actions[:h.n])
}

func (h *History) Len() int {
	return h.n
}

func (h *History) Get(i int) Action {
	if i >= h.n {
		panic(fmt.Errorf("index out of range: %d %v", i, h))
	}

	return h.actions[i]
}

func (h *History) Append(action Action) {
	if h.n >= len(h.actions) {
		panic(fmt.Errorf("history exceeded max capacity: %v", h))
	}

	h.actions[h.n] = action
	h.n++
}

// Gets the full, unabstracted infoset (but hashed into md5).
func (h *History) GetInfoSet(player Player, hand cards.Set) *InfoSet {
	return &InfoSet{
		History: h.asViewedBy(player),
		Hand:    hand,
	}
}

// Censor the given full game history to contain only info available
// to the given player.
func (h *History) asViewedBy(player Player) []EncodedAction {
	result := make([]EncodedAction, h.n)
	for i, action := range h.actions[:h.n] {
		if action.Player != player {
			// Hide the non-public information.
			action.PositionInDrawPile = 0
			action.CardsSeen = [3]cards.Card{}
		}

		result[i] = EncodeAction(action)
	}

	return result
}

type InfoSet struct {
	History []EncodedAction
	Hand    cards.Set
}

func (is *InfoSet) Key() string {
	var buf [3*MaxNumActions + 8]byte
	for i, action := range is.History {
		buf[i] = action[0]
		buf[i+1] = action[1]
		buf[i+2] = action[2]
	}

	// Player's hand is appended to private game history.
	binary.LittleEndian.PutUint64(buf[3*len(is.History):], uint64(is.Hand))

	// Hash into smaller bitstring since it is sparse.
	// We'll hope for no collisions :)
	hash := md5.Sum(buf[:3*len(is.History)+8])
	return string(hash[:])
}

type EncodedAction [3]uint8

// Action is packed as bits within a [3]uint8:
//   [0] Player (0 or 1)
//   [1-3] Type (1 - 4)
//   [4-7] Card (1 - 10)
//   [8-11] PositionInDrawPile (0 - 13)
//   [12-24] 3 Cards (1 - 10)
// Thus the first byte is public info, the second two bytes are private info.
func EncodeAction(a Action) EncodedAction {
	var result EncodedAction
	result[0] = uint8(a.Player)
	result[0] += uint8(a.Type << 1)
	result[0] += uint8(a.Card << 4)
	result[1] = uint8(a.PositionInDrawPile)
	result[1] += uint8(a.CardsSeen[0] << 4)
	result[2] = uint8(a.CardsSeen[1])
	result[2] += uint8(a.CardsSeen[2] << 4)
	return result
}

func (packed EncodedAction) Decode() Action {
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

func init() {
	gob.Register(&InfoSet{})
}
