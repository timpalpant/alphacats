package gamestate

import (
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
	InsertExplodingKitten
)

var allActions = []ActionType{
	DrawCard,
	PlayCard,
	GiveCard,
	InsertExplodingKitten,
}

var actionTypeStr = [...]string{
	"Invalid",
	"DrawCard",
	"PlayCard",
	"GiveCard",
	"InsertExplodingKitten",
}

func (t ActionType) String() string {
	return actionTypeStr[t]
}

// Action records each transition/edge in the game history.
type Action struct {
	Player Player
	Type   ActionType
	Card   cards.Card
	// When inserting the exploding kitten, this is the position that
	// the player chose to insert it to. NOTE: 1-based so that we can
	// distinguish no knowledge / random (0) from actual positions.
	PositionInDrawPile uint8         // Private information.
	CardsSeen          [3]cards.Card // Private information.
}

func (a Action) HasPrivateInfo() bool {
	return a.PositionInDrawPile != 0 ||
		a.CardsSeen[0] != 0 || a.CardsSeen[1] != 0 || a.CardsSeen[2] != 0
}

func (a Action) String() string {
	s := fmt.Sprintf("%s:%s", a.Player, a.Type)
	if a.Card != cards.Unknown {
		s += ":" + a.Card.String()
	}
	if a.Type == InsertExplodingKitten {
		if a.PositionInDrawPile == 0 {
			s += fmt.Sprintf(":RANDOM")
		} else {
			s += fmt.Sprintf(":%dth", a.PositionInDrawPile-1)
		}
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

const MaxNumActions = 58

// History records the history of game actions to reach this state.
// It is pre-sized to avoid allocations and keep GameState easily copyable.
type History struct {
	actions [MaxNumActions]EncodedAction
	n       int
}

func NewHistoryFromActions(actions []Action) History {
	h := History{}
	for _, action := range actions {
		h.Append(action)
	}
	return h
}

func (h *History) String() string {
	return fmt.Sprintf("%v", h.actions[:h.n])
}

func (h *History) Len() int {
	return h.n
}

func (h *History) Clear() {
	h.n = 0
}

func (h *History) Get(i int) Action {
	return h.GetPacked(i).Decode()
}

func (h *History) GetPacked(i int) EncodedAction {
	if i >= h.n {
		panic(fmt.Errorf("index out of range: %d %v", i, h))
	}

	return h.actions[i]
}

func (h *History) Append(action Action) {
	h.AppendPacked(EncodeAction(action))
}

func (h *History) AppendPacked(packed EncodedAction) {
	if h.n >= len(h.actions) {
		panic(fmt.Errorf("history exceeded max capacity: %v", h))
	}

	h.actions[h.n] = packed
	h.n++
}

func (h *History) Slice(n int) History {
	if n > h.Len() {
		panic(fmt.Errorf("attempting to slice %d actions of history with len=%d", n, h.Len()))
	}

	result := *h
	result.n = n
	return result
}

// Gets the current infoset for the given player.
func (h *History) GetInfoSet(player Player, hand cards.Set) InfoSet {
	return InfoSet{
		Player:  player,
		History: h.asViewedBy(player),
		Hand:    hand,
	}
}

func (h *History) AsSlice() []Action {
	result := make([]Action, h.Len())
	for i := range result {
		result[i] = h.Get(i)
	}
	return result
}

// Censor the given full game history to contain only info available
// to the given player.
func (h *History) asViewedBy(player Player) History {
	result := *h
	for i := 0; i < result.Len(); i++ {
		if player != h.actions[i].Player() {
			// Hide the non-public information.
			result.actions[i][1] = 0
			result.actions[i][2] = 0
		}
	}

	return result
}

type InfoSet struct {
	Player  Player
	History History
	Hand    cards.Set
}

// Key implements cfr.InfoSet.
func (is *InfoSet) Key() string {
	var buf [3 * MaxNumActions]byte
	bufSlice, _ := is.MarshalTo(buf[:])
	return string(bufSlice)
}

func (is *InfoSet) MarshalBinarySize() int {
	bufSize := 1 + 8 + is.History.Len()
	for i := 0; i < is.History.Len(); i++ {
		if is.History.actions[i].HasPrivateInfo() {
			bufSize += 2
		}
	}

	return bufSize
}

// MarshalBinary implements encoding.BinaryMarshaler.
func (is *InfoSet) MarshalBinary() ([]byte, error) {
	buf := make([]byte, 0, is.MarshalBinarySize())
	return is.MarshalTo(buf)
}

func (is *InfoSet) MarshalTo(buf []byte) ([]byte, error) {
	buf = append(buf, uint8(is.Player))
	for i := 0; i < is.History.Len(); i++ {
		action := is.History.actions[i]
		buf = append(buf, action[0])

		// Actions are "varint" encoded: we only copy the private bits
		// if they are non-zero, which is indicated by the last bit of
		// the first byte.
		if action.HasPrivateInfo() {
			buf = append(buf, action[1], action[2])
		}
	}

	// Player's hand is appended at the end.
	var hBuf [8]byte
	binary.LittleEndian.PutUint64(hBuf[:], uint64(is.Hand))
	buf = append(buf, hBuf[:]...)

	return buf, nil
}

// MarshalBinary implements encoding.BinaryUnmarshaler.
func (is *InfoSet) UnmarshalBinary(buf []byte) error {
	is.History.Clear()
	is.Player = Player(buf[0])
	buf = buf[1:]
	for len(buf) > 8 {
		packed := EncodedAction{}
		packed[0] = buf[0]
		buf = buf[1:]

		if packed.HasPrivateInfo() {
			packed[1] = buf[0]
			packed[2] = buf[1]
			buf = buf[2:]
		}

		is.History.AppendPacked(packed)
	}

	is.Hand = cards.Set(binary.LittleEndian.Uint64(buf))

	return nil
}

type EncodedAction [3]uint8

// Action is packed as bits within a [3]uint8:
//   [0] Player (0 or 1)
//   [1-2] Type (1 - 4, encoded as 0-3)
//   [3-6] Card (1 - 10)
//   [7] Indicates whether there is additional private info (remaining bits) (0 or 1)
//   [8-11] PositionInDrawPile (0 - 14)
//   [12-24] 3 Cards (1 - 10)
// Thus the first byte is public info, the second two bytes are private info.
func EncodeAction(a Action) EncodedAction {
	var result EncodedAction
	result[0] = uint8(a.Player)
	result[0] += uint8((a.Type - 1) << 1)
	result[0] += uint8(a.Card << 3)
	result[1] = uint8(a.PositionInDrawPile)
	result[1] += uint8(a.CardsSeen[0] << 4)
	result[2] = uint8(a.CardsSeen[1])
	result[2] += uint8(a.CardsSeen[2] << 4)
	hasPrivateInfo := (result[1] != 0 || result[2] != 0)
	if hasPrivateInfo {
		result[0] += uint8(1 << 7)
	}
	return result
}

func (packed EncodedAction) Decode() Action {
	return Action{
		Player:             Player(packed[0] & 0x1),
		Type:               ActionType((packed[0]>>1)&0x3) + 1,
		Card:               cards.Card((packed[0] >> 3) & 0xf),
		PositionInDrawPile: uint8(packed[1] & 0xf),
		CardsSeen: [3]cards.Card{
			cards.Card(packed[1] >> 4),
			cards.Card(packed[2] & 0xf),
			cards.Card(packed[2] >> 4),
		},
	}
}

func (packed EncodedAction) Player() Player {
	return Player(packed[0] & 0x1)
}

func (packed EncodedAction) HasPrivateInfo() bool {
	return (packed[0] >> 7) == 1
}

func EncodeActions(actions []Action) []EncodedAction {
	result := make([]EncodedAction, len(actions))
	for i, action := range actions {
		result[i] = EncodeAction(action)
	}

	return result
}

func init() {
	gob.Register(&InfoSet{})
}
