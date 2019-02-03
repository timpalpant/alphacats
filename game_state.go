package alphacats

import (
	"fmt"

	"github.com/pkg/errors"

	"github.com/timpalpant/alphacats/cards"
)

type Player uint8

const (
	Player0 Player = iota
	Player1
)

var playerStr = [...]string{
	"Player0",
	"Player1",
}

func (p Player) String() string {
	return playerStr[p]
}

type TurnType int

const (
	DrawCard TurnType = iota
	Deal
	PlayTurn
	GiveCard
	MustDefuse
	SeeTheFuture
	GameOver
)

var turnTypeStr = [...]string{
	"DrawCard",
	"Deal",
	"PlayTurn",
	"GiveCard",
	"MustDefuse",
	"SeeTheFuture",
	"GameOver",
}

func (tt TurnType) IsChance() bool {
	return tt == DrawCard || tt == Deal || tt == SeeTheFuture
}

func (tt TurnType) String() string {
	return turnTypeStr[tt]
}

// GameState represents the current state of the game.
type GameState struct {
	// Set of Cards remaining in the draw pile.
	DrawPile cards.Set
	// Info observable from the point of view of either player.
	Player0Info InfoSet
	Player1Info InfoSet
}

// Validate sanity checks the GameState to ensure we have maintained
// internal consistency in the game tree.
func (gs *GameState) Validate() error {
	if err := gs.Player0Info.Validate(); err != nil {
		return errors.Wrapf(err, "player %v info invalid", Player0)
	}

	if err := gs.Player1Info.Validate(); err != nil {
		return errors.Wrapf(err, "player %v info invalid", Player1)
	}

	// All player's should know the current number of cards in the draw pile.
	if gs.DrawPile.Len() != gs.Player0Info.NumCardsInDrawPile() {
		return fmt.Errorf("player %v thinks draw pile has %d cards, actually %d",
			Player0, gs.Player0Info.NumCardsInDrawPile(), gs.DrawPile.Len())
	}
	if gs.DrawPile.Len() != gs.Player1Info.NumCardsInDrawPile() {
		return fmt.Errorf("player %v thinks draw pile has %d cards, actually %d",
			Player1, gs.Player1Info.NumCardsInDrawPile(), gs.DrawPile.Len())
	}

	// All players should have the same view of the discard pile.
	if gs.Player0Info.DiscardPile != gs.Player1Info.DiscardPile {
		return fmt.Errorf("players do not agree on discard pile: %v != %v",
			gs.Player0Info.DiscardPile, gs.Player1Info.DiscardPile)
	}

	// All fixed draw pile cards must be in the draw pile.
	for i := 0; i < gs.DrawPile.Len(); i++ {
		card := gs.FixedDrawPileCards().NthCard(i)
		if card != cards.Unknown && gs.DrawPile.CountOf(card) == 0 {
			return fmt.Errorf("card %v fixed at position %v in draw pile but not in set %v",
				card, i, gs.DrawPile)
		}
	}

	// If a draw pile card is fixed in the view of either player,
	// it must be fixed here as well.
	for i := 0; i < gs.DrawPile.Len(); i++ {
		p0Card := gs.Player0Info.KnownDrawPileCards.NthCard(i)
		if p0Card != cards.Unknown && p0Card != gs.FixedDrawPileCards().NthCard(i) {
			return fmt.Errorf("player %v thinks draw pile position %d is %v, but actually %v",
				Player0, i, p0Card, gs.FixedDrawPileCards().NthCard(i))
		}

		p1Card := gs.Player1Info.KnownDrawPileCards.NthCard(i)
		if p1Card != cards.Unknown && p1Card != gs.FixedDrawPileCards().NthCard(i) {
			return fmt.Errorf("player %v thinks draw pile position %d is %v, but actually %v",
				Player1, i, p1Card, gs.FixedDrawPileCards().NthCard(i))
		}
	}

	return nil
}

// FixedDrawPileCards is the union of all cards in the draw pile whose identity
// is fixed because either one of the players knows it.
func (gs *GameState) FixedDrawPileCards() cards.Stack {
	result := gs.Player0Info.KnownDrawPileCards
	for i := 0; i < gs.DrawPile.Len(); i++ {
		card := gs.Player1Info.KnownDrawPileCards.NthCard(i)
		if card != cards.Unknown {
			p0View := result.NthCard(i)
			if p0View != cards.Unknown && p0View != card {
				panic(fmt.Errorf("inconsistent fixed card info: p0 = %v, p1 = %v",
					p0View, card))
			}

			result.SetNthCard(i, card)
		}
	}

	return result
}

func (gs *GameState) InfoSet(p Player) *InfoSet {
	if p == Player0 {
		return &gs.Player0Info
	}

	return &gs.Player1Info
}

func (gs *GameState) GetPlayerHand(p Player) cards.Set {
	return gs.InfoSet(p).OurHand
}
