package gamestate

import (
	"fmt"

	"github.com/pkg/errors"

	"github.com/timpalpant/alphacats/cards"
)

var (
	fixedCardProbabilities = make([]map[cards.Card]float64, cards.Cat+1)
	cardProbabilitiesCache = map[cards.Set]map[cards.Card]float64{}
)

func init() {
	for card := cards.Card(0); card <= cards.Cat; card++ {
		fixedCardProbabilities[card] = map[cards.Card]float64{card: 1.0}
	}
}

// GameState represents the current state of the game.
//
// Any additional fields added to GameState must also be added to clone().
type GameState struct {
	// The history of player actions that were taken to reach this state.
	history publicHistory
	// Set of Cards remaining in the draw pile.
	// Note that the players will not in general have access to this information.
	drawPile cards.Set
	// Cards in the draw pile whose identity is fixed because one of the player's
	// knows it.
	fixedDrawPileCards cards.Stack
	// The Cards that have been played in the discard pile.
	discardPile cards.Stack
	// Private information held from the point of view of either player.
	player0Info privateInfo
	player1Info privateInfo
}

// New returns a new GameState created by dealing the given sets of cards
// to each player at the beginning of the game.
func New(player0Deal, player1Deal cards.Set) GameState {
	remainingCards := cards.CoreDeck
	remainingCards.RemoveAll(player0Deal)
	remainingCards.RemoveAll(player1Deal)
	remainingCards.Add(cards.ExplodingCat)
	return GameState{
		drawPile:    remainingCards,
		player0Info: newPrivateInfo(player0Deal),
		player1Info: newPrivateInfo(player1Deal),
	}
}

// Apply returns the new GameState created by applying the given Action.
func Apply(state GameState, action Action) GameState {
	switch action.Type {
	case DrawCard:
		state.drawCard(action.Player, action.Card, action.PositionInDrawPile)
	case PlayCard:
		state.playCard(action.Player, action.Card)
	case GiveCard:
		state.giveCard(action.Player, action.Card)
	case InsertExplodingCat:
		state.insertExplodingCat(action.Player, action.PositionInDrawPile)
	case SeeTheFuture:
		state.seeTopNCards(action.Player, action.Cards)
	default:
		panic(fmt.Errorf("invalid action: %+v", action))
	}

	state.history.Append(action)
	return state
}

func (gs *GameState) String() string {
	return fmt.Sprintf("draw pile: %s, fixed: %s, discard: %s, p0: %s, p1: %s",
		gs.drawPile, gs.fixedDrawPileCards,
		gs.discardPile, gs.player0Info.String(), gs.player1Info.String())
}

// Validate sanity checks the GameState to ensure we have maintained
// internal consistency in the game tree.
func (gs *GameState) Validate() error {
	if err := gs.player0Info.validate(); err != nil {
		return errors.Wrapf(err, "player %v info invalid", Player0)
	}

	if err := gs.player1Info.validate(); err != nil {
		return errors.Wrapf(err, "player %v info invalid", Player1)
	}

	// Both players should know the correct total number of cards in each other's hand.
	if gs.player0Info.opponentHand.Len() != gs.player1Info.ourHand.Len() {
		return fmt.Errorf("Player %v thinks %v has %d cards, but they have %d",
			Player0, Player1, gs.player0Info.opponentHand.Len(), gs.player1Info.ourHand.Len())
	}
	if gs.player1Info.opponentHand.Len() != gs.player0Info.ourHand.Len() {
		return fmt.Errorf("Player %v thinks %v has %d cards, but they have %d",
			Player1, Player0, gs.player1Info.opponentHand.Len(), gs.player0Info.ourHand.Len())
	}

	// All fixed draw pile cards must be in the draw pile.
	for i := 0; i < gs.drawPile.Len(); i++ {
		card := gs.fixedDrawPileCards.NthCard(i)
		if card != cards.Unknown && !gs.drawPile.Contains(card) {
			return fmt.Errorf("card %v fixed at position %v in draw pile but not in set %v",
				card, i, gs.drawPile)
		}
	}

	// Players must not have fixed card info that contradicts reality.
	p0Known := gs.player0Info.effectiveKnownDrawPileCards()
	p1Known := gs.player1Info.effectiveKnownDrawPileCards()
	for i := 0; i < gs.drawPile.Len(); i++ {
		card := gs.fixedDrawPileCards.NthCard(i)
		p0Card := p0Known.NthCard(i)
		if p0Card != cards.Unknown && p0Card != card {
			return fmt.Errorf("player %v thinks %dth card is %v but is actually %v",
				Player0, i, p0Card, card)
		}

		p1Card := p1Known.NthCard(i)
		if p1Card != cards.Unknown && p1Card != card {
			return fmt.Errorf("player %v thinks %dth card is %v but is actually %v",
				Player1, i, p1Card, card)
		}
	}

	// If a player knows a card in the other player's hand, they must
	// actually have it. Note: They may have more (that are Unknown top opponent).
	p0Unknown := gs.player0Info.opponentHand.CountOf(cards.Unknown)
	p1Unknown := gs.player1Info.opponentHand.CountOf(cards.Unknown)
	for card := cards.Unknown + 1; card <= cards.Cat; card++ {
		n := gs.player0Info.opponentHand.CountOf(card)
		m := gs.player1Info.ourHand.CountOf(card)
		if m < n || m > n+p0Unknown {
			return fmt.Errorf("player 0 thinks player 1 has %d of %v, but they actually have %d",
				n, card, m)
		}

		n = gs.player1Info.opponentHand.CountOf(card)
		m = gs.player0Info.ourHand.CountOf(card)
		if m < n || m > n+p1Unknown {
			return fmt.Errorf("player 1 thinks player 0 has %d of %v, but they actually have %d",
				n, card, m)
		}
	}

	return nil
}

func (gs *GameState) GetHistory() []Action {
	return gs.history.AsSlice()
}

func (gs *GameState) GetDrawPile() cards.Set {
	return gs.drawPile
}

func (gs *GameState) GetPlayerHand(p Player) cards.Set {
	return gs.privateInfo(p).ourHand
}

func (gs *GameState) HasDefuseCard(p Player) bool {
	return gs.GetPlayerHand(p).Contains(cards.Defuse)
}

// InfoSet represents the state of the game from the point of view of one of the
// players. Note that multiple distinct game states may have the same InfoSet
// due to hidden information that the player is not privy to.
type InfoSet struct {
	public  publicHistory
	private privateInfo
}

func (gs *GameState) GetInfoSet(player Player) InfoSet {
	return InfoSet{
		private: *gs.privateInfo(player),
		public:  gs.history,
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

	return toProbabilities(candidates)
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

	return toProbabilities(candidates)
}

func toProbabilities(candidates cards.Set) map[cards.Card]float64 {
	if result, ok := cardProbabilitiesCache[candidates]; ok {
		return result
	}

	counts := candidates.Counts()
	result := make(map[cards.Card]float64, len(counts))
	nTotal := float64(candidates.Len())
	for card, count := range counts {
		p := float64(count) / nTotal
		result[card] = p
	}

	cardProbabilitiesCache[candidates] = result
	return result
}

func (gs *GameState) privateInfo(p Player) *privateInfo {
	if p == Player0 {
		return &gs.player0Info
	}

	return &gs.player1Info
}

func (gs *GameState) playCard(player Player, card cards.Card) {
	gs.privateInfo(player).playCard(card)
	gs.privateInfo(1 - player).opponentPlayedCard(card)
	gs.discardPile.InsertCard(card, 0)
}

func (gs *GameState) drawCard(player Player, card cards.Card, position int) {
	// Pop card from the draw pile.
	gs.drawPile.Remove(card)
	gs.fixedDrawPileCards.RemoveCard(position)
	// Add to player's hand.
	gs.privateInfo(player).drawCard(card, position)
	gs.privateInfo(1-player).opponentDrewCard(card, position)
}

func (gs *GameState) insertExplodingCat(player Player, position int) {
	// Place exploding cat card in the Nth position in draw pile.
	gs.drawPile.Add(cards.ExplodingCat)
	gs.fixedDrawPileCards.InsertCard(cards.ExplodingCat, position)
	gs.privateInfo(player).ourHand.Remove(cards.ExplodingCat)
	gs.privateInfo(player).knownDrawPileCards.InsertCard(cards.ExplodingCat, position)
	gs.privateInfo(1 - player).opponentHand.Remove(cards.ExplodingCat)
	gs.privateInfo(1 - player).pendingKittenInterruption = true
}

func (gs *GameState) seeTopNCards(player Player, topN []cards.Card) {
	gs.privateInfo(player).seeTopCards(topN)

	for i, card := range topN {
		nthCard := gs.fixedDrawPileCards.NthCard(i)
		if nthCard != cards.Unknown && nthCard != card {
			panic(fmt.Errorf("we knew %d th card to be %v, but are now told it is %v",
				i, nthCard, card))
		}

		gs.fixedDrawPileCards.SetNthCard(i, card)
	}
}

func (gs *GameState) giveCard(player Player, card cards.Card) {
	pInfo := gs.privateInfo(player)
	pInfo.ourHand.Remove(card)
	pInfo.opponentHand.Add(card)

	opponentInfo := gs.privateInfo(1 - player)
	opponentInfo.ourHand.Add(card)
	if opponentInfo.opponentHand.CountOf(card) > 0 {
		// If opponent already knew we had one of these cards
		opponentInfo.opponentHand.Remove(card)
	} else {
		// Otherwise it was one of the Unknown cards in our hand.
		opponentInfo.opponentHand.Remove(cards.Unknown)
	}
}
