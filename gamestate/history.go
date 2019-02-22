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
		s += fmt.Sprintf(":%s", a.CardsSeen[0])
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
	actions [MaxNumActions]Action
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

	return h.actions[i]
}

func (h *history) Append(action Action) {
	if h.n >= len(h.actions) {
		panic(fmt.Errorf("history exceeded max length: %v", h))
	}

	h.actions[h.n] = action
	h.n++
}

func (h *history) AsSlice() []Action {
	return h.actions[:h.n]
}

// Gets abstracted infoset constructed for the given player.
func (h *history) GetAbstractedInfoSet(player Player, hand cards.Set, nRemaining int) string {
	// Build abstracted info set by calculating and appending:
	//   1. Set of cards in our hand
	//   2. Stack of cards in discard pile
	//   3. Set of cards (some may be unknown) in opponent's hand
	//   4. Next three cards (if we know them).
	//   5. Number of cards remaining in draw pile
	//   6. Position of exploding kitten (if we know it).
	var buf [20]byte
	binary.LittleEndian.PutUint64(buf[0:], uint64(hand))
	discardPile := h.getDiscardPile()
	binary.LittleEndian.PutUint64(buf[8:], uint64(discardPile.ToSet()))
	next3 := h.getNext3Cards(player)
	buf[16] = uint8(next3[0])
	buf[17] = uint8(next3[1])
	buf[18] = uint8(next3[1])
	buf[19] = uint8(nRemaining)
	return string(buf[:])
}

func (h *history) getNext3Cards(player Player) [3]cards.Card {
	var result [3]cards.Card
	offsetKnownExploding := -1
	for i := 0; i < h.n; i++ {
		action := h.actions[i]
		if action.Player == player {
			if action.Type == PlayCard && action.Card == cards.SeeTheFuture {
				copy(result[:], action.CardsSeen[:])
			} else if action.Type == InsertExplodingCat {
				offsetKnownExploding = int(action.PositionInDrawPile)
			}
		}

		if action.Type == DrawCard {
			result[0] = result[1]
			result[1] = result[2]
			result[2] = cards.Unknown
			offsetKnownExploding--
		} else if action.Type == PlayCard && action.Card == cards.Shuffle {
			result = [3]cards.Card{}
			offsetKnownExploding = -1
		} else if action.Player != player && action.Type == InsertExplodingCat {
			// FIXME: This could be less aggressive. When the opponent
			// inserts the exploding kitten it doesn't completely erase our knowledge.
			result = [3]cards.Card{}
			offsetKnownExploding = -1
		}
	}

	if offsetKnownExploding >= 0 && offsetKnownExploding < 3 {
		result[offsetKnownExploding] = cards.ExplodingCat
	}

	return result
}

func (h *history) getDiscardPile() cards.Stack {
	discardPile := cards.NewStack()
	for i := 0; i < h.n; i++ {
		if h.actions[i].Type == PlayCard {
			discardPile.InsertCard(h.actions[i].Card, 0)
		}
	}

	return discardPile
}

// Gets the full, unabstracted infoset (but hashed into md5).
func (h *history) GetInfoSet(player Player, hand cards.Set) string {
	var buf [3*MaxNumActions + 8]byte
	for i := 0; i < h.n; i++ {
		action := h.actions[i]
		packed := encodeAction(action)
		buf[i] = packed[0]
		if action.Player == player {
			buf[i+1] = packed[1]
			buf[i+2] = packed[2]
		}
	}

	// Player's hand is appended to private game history.
	binary.LittleEndian.PutUint64(buf[3*h.n:], uint64(hand))

	// Hash into smaller bitstring since it is sparse.
	// We'll hope for no collisions :)
	hash := md5.Sum(buf[:3*h.n+8])
	return string(hash[:])
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
