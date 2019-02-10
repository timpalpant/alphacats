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
