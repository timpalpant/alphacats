package gamestate

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/timpalpant/alphacats/cards"
)

// GameState represents the current state of the game.
//
// Any additional fields added to GameState must also be added to clone().
type GameState struct {
	// The history of player actions that were taken to reach this state.
	history history
	// Set of Cards remaining in the draw pile.
	// Note that the players will not in general have access to this information.
	drawPile cards.Set
	// Cards in the draw pile whose identity is fixed because one of the player's
	// knows it.
	fixedDrawPileCards cards.Stack
	player0Hand        cards.Set
	player1Hand        cards.Set
}

// New returns a new GameState created by dealing the given sets of cards
// to each player at the beginning of the game.
func New(player0Deal, player1Deal cards.Set) GameState {
	remainingCards := cards.CoreDeck
	remainingCards.RemoveAll(player0Deal)
	remainingCards.RemoveAll(player1Deal)
	remainingCards.Add(cards.ExplodingCat)
	player0Deal.Add(cards.Defuse)
	player1Deal.Add(cards.Defuse)
	return GameState{
		drawPile:    remainingCards,
		player0Hand: player0Deal,
		player1Hand: player1Deal,
	}
}

// Apply returns the new GameState created by applying the given Action.
func (gs *GameState) Apply(action Action) {
	switch action.Type {
	case PlayCard:
	case DrawCard:
		gs.drawCard(action.Player, action.Card, int(action.PositionInDrawPile))
	case GiveCard:
		gs.giveCard(action.Player, action.Card)
	case InsertExplodingCat:
		gs.insertExplodingCat(action.Player, int(action.PositionInDrawPile))
	case SeeTheFuture:
		gs.seeTop3Cards(action.Player, action.Cards)
	default:
		panic(fmt.Errorf("invalid action: %+v", action))
	}

	gs.history.Append(action)
}

func (gs *GameState) String() string {
	return fmt.Sprintf("draw pile: %s, fixed: %s, p0: %s, p1: %s",
		gs.drawPile, gs.fixedDrawPileCards,
		gs.player0Hand, gs.player1Hand)
}

func (gs *GameState) GetHistory() []Action {
	return gs.history.AsSlice()
}

func (gs *GameState) GetDrawPile() cards.Set {
	return gs.drawPile
}

func (gs *GameState) GetPlayerHand(p Player) cards.Set {
	if p == Player0 {
		return gs.player0Hand
	}

	return gs.player1Hand
}

func (gs *GameState) HasDefuseCard(p Player) bool {
	return gs.GetPlayerHand(p).Contains(cards.Defuse)
}

func (gs *GameState) LastActionWasSlap() bool {
	lastAction := gs.history.Get(gs.history.Len() - 1)
	return lastAction.Type == PlayCard && (lastAction.Card == cards.Slap1x || lastAction.Card == cards.Slap2x)
}

// InfoSet represents the state of the game from the point of view of one of the
// players. Note that multiple distinct game states may have the same InfoSet
// due to hidden information that the player is not privy to.
type InfoSet struct {
	history history
	hand    cards.Set
}

func (is *InfoSet) String() string {
	var builder strings.Builder
	if err := binary.Write(&builder, binary.LittleEndian, is.hand); err != nil {
		panic(err)
	}

	for _, action := range is.history.AsSlice() {
		packed := encodeAction(action)
		if err := binary.Write(&builder, binary.LittleEndian, packed); err != nil {
			panic(err)
		}
	}

	return builder.String()
}

func (gs *GameState) GetInfoSet(player Player) InfoSet {
	return InfoSet{
		history: gs.history.GetPlayerView(player),
		hand:    gs.GetPlayerHand(player),
	}
}

func (gs *GameState) BottomCardProbabilities() map[cards.Card]float64 {
	bottom := gs.drawPile.Len() - 1
	bottomCard := gs.fixedDrawPileCards.NthCard(bottom)
	if bottomCard != cards.Unknown {
		// Identity of the bottom card is fixed.
		return fixedCardProbabilities[bottomCard]
	}

	// Note: We need to exclude any cards whose identity is already fixed in a
	// position known *not* to be the bottom card.
	start := 0
	end := gs.drawPile.Len() - 1
	candidates := gs.drawPile
	for i := start; i < end; i++ {
		if known := gs.fixedDrawPileCards.NthCard(i); known != cards.Unknown {
			candidates.Remove(known)
		}
	}

	result, ok := cardProbabilitiesCache[candidates]
	if !ok {
		panic(fmt.Errorf("missing card probabilities for: %v", candidates))
	}

	return result
}

func (gs *GameState) TopCardProbabilities() map[cards.Card]float64 {
	topCard := gs.fixedDrawPileCards.NthCard(0)
	if topCard != cards.Unknown {
		return fixedCardProbabilities[topCard]
	}

	// Note: We need to exclude any cards whose identity is already fixed in a
	// position known *not* to be the top card.
	start := 1
	end := gs.drawPile.Len()

	candidates := gs.drawPile
	for i := start; i < end; i++ {
		if known := gs.fixedDrawPileCards.NthCard(i); known != cards.Unknown {
			candidates.Remove(known)
		}
	}

	result, ok := cardProbabilitiesCache[candidates]
	if !ok {
		panic(fmt.Errorf("missing card probabilities for: %v", candidates))
	}

	return result
}

func (gs *GameState) drawCard(player Player, card cards.Card, position int) {
	// Pop card from the draw pile.
	gs.drawPile.Remove(card)
	gs.fixedDrawPileCards.RemoveCard(position)
}

func (gs *GameState) insertExplodingCat(player Player, position int) {
	// Place exploding cat card in the Nth position in draw pile.
	gs.drawPile.Add(cards.ExplodingCat)
	gs.fixedDrawPileCards.InsertCard(cards.ExplodingCat, position)
}

func (gs *GameState) seeTop3Cards(player Player, top3 [3]cards.Card) {
	for i, card := range top3 {
		nthCard := gs.fixedDrawPileCards.NthCard(i)
		if nthCard != cards.Unknown && nthCard != card {
			panic(fmt.Errorf("we knew %d th card to be %v, but are now told it is %v",
				i, nthCard, card))
		}

		gs.fixedDrawPileCards.SetNthCard(i, card)
	}
}

func (gs *GameState) giveCard(player Player, card cards.Card) {
	if player == Player0 {
		gs.player0Hand.Remove(card)
		gs.player1Hand.Add(card)
	} else {
		gs.player1Hand.Remove(card)
		gs.player0Hand.Add(card)
	}
}
