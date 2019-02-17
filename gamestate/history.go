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
	DrawCardFromBottom
	PlayCard
	GiveCard
	InsertExplodingCat
	SeeTheFuture
)

var allActions = []ActionType{
	DrawCard,
	DrawCardFromBottom,
	PlayCard,
	GiveCard,
	InsertExplodingCat,
	SeeTheFuture,
}

var actionTypeStr = [...]string{
	"Invalid",
	"DrawCard",
	"DrawCardFromBottom",
	"PlayCard",
	"GiveCard",
	"InsertExplodingCat",
	"SeeTheFuture",
}

func (t ActionType) IsPrivate() bool {
	return t == DrawCard || t == InsertExplodingCat || t == SeeTheFuture
}

func (t ActionType) String() string {
	return actionTypeStr[t]
}

// Action records each transition/edge in the game history.
type Action struct {
	Player             Player
	Type               ActionType
	Card               cards.Card    // May be private information.
	PositionInDrawPile int           // May be private information.
	Cards              [3]cards.Card // May be private information.
}

// Action is packed as bits within a uint32:
// [0] Player
// [1-3] Type
// [4-7] Card
// [8-11] PositionInDrawPile (0 - 13)
// [12-24] 3 Cards
func encodeAction(a Action) uint32 {
	result := uint32(a.Player)
	result += uint32(a.Type << 1)
	result += uint32(a.Card << 4)
	result += uint32(a.PositionInDrawPile << 8)
	for i := uint(0); i < 3; i++ {
		shift := 4*i + 12
		result += uint32(a.Cards[i] << shift)
	}
	return result
}

func decodeAction(packed uint32) Action {
	action := Action{
		Player:             Player(packed & 0x1),
		Type:               ActionType((packed >> 1) & 0x7),
		Card:               cards.Card((packed >> 4) & 0xf),
		PositionInDrawPile: int((packed >> 8) & 0xf),
	}

	for i := uint(0); i < 3; i++ {
		shift := 4*i + 12
		action.Cards[i] = cards.Card((packed >> shift) & 0xf)
	}

	return action
}

func (a Action) String() string {
	s := fmt.Sprintf("%s:%s", a.Player, a.Type)
	if a.Card != cards.Unknown {
		s += ":" + a.Card.String()
	}
	if a.Type == InsertExplodingCat {
		s += fmt.Sprintf(":%d", a.PositionInDrawPile)
	}
	if a.Type == SeeTheFuture {
		s += fmt.Sprintf(":%v", a.Cards)
	}
	return s
}

const MaxNumActions = 48

// History records the history of game actions to reach this state.
// It is pre-sized to avoid allocations.
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

func (h *history) GetPlayerView(p Player) history {
	result := history{}
	for i := 0; i < h.n; i++ {
		action := h.actions[i]
		if action.Player != p && action.Type.IsPrivate() {
			action.PositionInDrawPile = 0
			action.Card = cards.Unknown
			action.Cards = [3]cards.Card{}
		}

		result.Append(action)
	}
	return result
}

func (h *history) AsSlice() []Action {
	return h.actions[:h.n]
}
