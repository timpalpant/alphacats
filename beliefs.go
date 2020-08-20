package alphacats

import (
	"fmt"
	"math/rand"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

// Core deck - 2x player hands + Defuse + Exploding Kitten
var initialNumCardsInDrawPile = cards.CoreDeck.Len() - 2*4 + 2

// BeliefState holds the distribution of probabilities over underlying
// game states as perceived from the point of view of one player.
type BeliefState struct {
	opponentPolicy func(cfr.GameTreeNode) []float32
	infoSet        gamestate.InfoSet
	states         []*GameNode
	reachProbs     []float32
}

// Return all game states consistent with the given initial hand.
// Note that the passed hand should include the Defuse card.
func NewBeliefState(opponentPolicy func(cfr.GameTreeNode) []float32, infoSet gamestate.InfoSet) *BeliefState {
	tbdDrawPile := cards.NewStack()
	for i := 0; i < initialNumCardsInDrawPile; i++ {
		tbdDrawPile.SetNthCard(i, cards.TBD)
	}

	remaining := cards.CoreDeck
	privateDeal := infoSet.Hand
	privateDeal.Remove(cards.Defuse)
	remaining.RemoveAll(privateDeal)

	var states []*GameNode
	seen := make(map[cards.Set]struct{})
	enumerateDealsHelper(remaining, cards.NewSet(), privateDeal.Len(), func(opponentDeal cards.Set) {
		if _, ok := seen[opponentDeal]; ok {
			return
		}

		seen[opponentDeal] = struct{}{}
		var p0Deal, p1Deal cards.Set
		if infoSet.Player == gamestate.Player0 {
			p0Deal = privateDeal
			p1Deal = opponentDeal
		} else {
			p0Deal = opponentDeal
			p1Deal = privateDeal
		}

		p0Deal.Add(cards.Defuse)
		p1Deal.Add(cards.Defuse)
		game := NewGame(tbdDrawPile, p0Deal, p1Deal)
		states = append(states, game)
	})

	return &BeliefState{
		opponentPolicy: opponentPolicy,
		infoSet:        infoSet,
		states:         states,
		reachProbs:     uniformDistribution(len(states)),
	}
}

func (bs *BeliefState) Len() int {
	return len(bs.states)
}

func (bs *BeliefState) Less(i, j int) bool {
	return bs.reachProbs[i] < bs.reachProbs[j]
}

func (bs *BeliefState) Swap(i, j int) {
	bs.states[i], bs.states[j] = bs.states[j], bs.states[i]
	bs.reachProbs[i], bs.reachProbs[j] = bs.reachProbs[j], bs.reachProbs[i]
}

// Update belief state by propagating all current states forward,
// expanding determinizations as necessary and filtering to those that match
// the given new info set.
func (bs *BeliefState) Update(infoSet gamestate.InfoSet) {
	if bs.states[0].Type() == cfr.ChanceNodeType {
		bs.updateChanceAction(infoSet)
	} else if infoSet.Player == bs.infoSet.Player {
		bs.updateSelfAction(infoSet)
	} else {
		bs.updateOpponentAction(infoSet)
	}

	glog.V(2).Infof("Belief state now has %d states", len(bs.states))
	bs.infoSet = infoSet
}

type weightedBelief struct {
	node *GameNode
	p    float32
}

type shuffledState struct {
	drawPile      cards.Set
	hand          cards.Set
	publicHistory gamestate.History
}

func (bs *BeliefState) updateChanceAction(infoSet gamestate.InfoSet) {
	// Optimization: collapse all states that are now equivalent.
	// Any previous knowledge about the specific ordering of cards in the draw pile
	// is now irrelevant (although knowing the existence of particular cards in the
	// pile is still relevant).
	seen := make(map[shuffledState]weightedBelief)
	for i, state := range bs.states {
		if state.Type() != cfr.ChanceNodeType {
			panic(fmt.Errorf("Updating beliefs as if at a chance node, but belief state is a %v", state.Type()))
		}

		is := state.GetInfoSet(1 - infoSet.Player)
		var publicHistory gamestate.History
		for j := 0; j < is.History.Len(); j++ {
			action := is.History.Get(j)
			action.PositionInDrawPile = 0
			action.CardsSeen = [3]cards.Card{}
			publicHistory.Append(action)
		}
		shuffleState := shuffledState{
			drawPile:      state.GetDrawPile().ToSet(),
			hand:          is.Hand,
			publicHistory: publicHistory,
		}

		// NOTE: Since, after the initial deal, the only chance action in the
		// game is when the Shuffle card is played, we can implement it by simply
		// clearing all knowledge of the draw pile state.
		b := seen[shuffleState]
		child := state.GetChild(0).(*GameNode)
		shuffledState := clearDrawPileKnowledge(child.GetState())
		b.node = child.CloneWithState(shuffledState)
		b.p += bs.reachProbs[i]
		seen[shuffleState] = b
		state.Close()
	}

	bs.states = bs.states[:0]
	bs.reachProbs = bs.reachProbs[:0]
	for _, b := range seen {
		bs.states = append(bs.states, b.node)
		bs.reachProbs = append(bs.reachProbs, b.p)
	}
}

func (bs *BeliefState) updateSelfAction(infoSet gamestate.InfoSet) {
	if infoSet.History.Len() == bs.infoSet.History.Len() {
		// This happens when the exploding kitten was just picked up.
		// The game state does not advance (because we just drew a card,
		// but the game tree advances to an InsertExplodingKitten node.
		return
	} else if infoSet.History.Len() < bs.infoSet.History.Len() {
		panic(fmt.Errorf("Got info set with less history than previous info set: %+v vs. %+v", infoSet, bs.infoSet))
	}

	action := infoSet.History.Get(bs.infoSet.History.Len())
	bs.determinizeForAction(action)
	var newStates []*GameNode
	var newReachProbs []float32
	nChildren := 0
	for i, determinization := range bs.states {
		for j := 0; j < determinization.NumChildren(); j++ {
			nChildren++
			child := determinization.GetChild(j).(*GameNode)
			is := child.GetInfoSet(bs.infoSet.Player)
			if is == infoSet {
				// Determinized game is consistent with our observed history.
				newStates = append(newStates, child.Clone())
				newReachProbs = append(newReachProbs, bs.reachProbs[i])
			}
		}

		determinization.Close()
	}

	if len(newStates) == 0 {
		glog.Errorf("Old info set: hand: %s, history: %s", bs.infoSet.Hand, bs.infoSet.History)
		glog.Errorf("New info set: hand: %s, history: %s", infoSet.Hand, infoSet.History)
		glog.Infof("Children considered: %d", nChildren)
		glog.Infof("States considered: %s", len(bs.states))
		panic(fmt.Errorf("Belief state is empty!"))
	}

	bs.states = newStates
	bs.reachProbs = newReachProbs
}

func (bs *BeliefState) updateOpponentAction(infoSet gamestate.InfoSet) {
	action := infoSet.History.Get(bs.infoSet.History.Len())
	bs.determinizeForAction(action)
	var newStates []*GameNode
	var newReachProbs []float32
	for i, determinization := range bs.states {
		policyP := bs.opponentPolicy(determinization)
		for j := 0; j < determinization.NumChildren(); j++ {
			child := determinization.GetChild(j).(*GameNode)
			is := child.GetInfoSet(gamestate.Player1)
			if is == infoSet {
				// Determinized game is consistent with our observed history.
				newStates = append(newStates, child.Clone())
				newReachProbs = append(newReachProbs, policyP[j]*bs.reachProbs[i])
			}
		}

		determinization.Close()
	}

	bs.states = newStates
	bs.reachProbs = newReachProbs
}

func (bs *BeliefState) determinizeForAction(action gamestate.Action) {
	glog.V(2).Infof("Determinizing for action: %v", action)
	// Determinize just enough info so that all actions are fully specified.
	switch action.Type {
	case gamestate.PlayCard:
		if action.Card == cards.SeeTheFuture {
			if action.CardsSeen[0] != cards.Unknown {
				// We know the specific cards that were seen, so we can avoid
				// fully expanding the set of possibiliies.
				bs.determinizeSeenCards(action.CardsSeen)
			} else {
				bs.determinizeTopKCards(3)
			}
		} else if action.Card == cards.DrawFromTheBottom {
			drawnCard := action.CardsSeen[0]
			if drawnCard != cards.Unknown {
				// We know the specific card that was drawn, so we can avoid
				// fully expanding the set of possibiliies.
				bs.determinizeDrawnCardFromBottom(drawnCard)
			} else {
				bs.determinizeForDrawFromTheBottom()
			}
		}
	case gamestate.DrawCard:
		drawnCard := action.CardsSeen[0]
		if drawnCard != cards.Unknown {
			// We know the specific card that was drawn, so we can avoid
			// fully expanding the set of possibiliies.
			bs.determinizeDrawnCard(drawnCard)
		} else {
			bs.determinizeTopKCards(1)
		}
	}
}

func (bs *BeliefState) determinizeSeenCards(seenCards [3]cards.Card) {
	var newStates []*GameNode
	var newReachProbs []float32
	for i, game := range bs.states {
		state := game.GetState()
		drawPile := state.GetDrawPile()
		incompatibleState := false
		for i, card := range seenCards {
			drawPileCard := drawPile.NthCard(i)
			if drawPileCard == cards.TBD {
				freeCards := getFreeCards(state)
				if !freeCards.Contains(card) {
					// This state could not possibly be valid because we drew a card
					// that was known not to be among the set of undetermined cards.
					incompatibleState = true
					break
				}

				drawPile.SetNthCard(i, card)
				state = gamestate.NewShuffled(state, drawPile)
			} else if drawPileCard != card {
				incompatibleState = true
				break
			} // else: card was already determined to be the seen card.
		}

		if incompatibleState {
			continue
		}

		determinizedGame := game.CloneWithState(state)
		newStates = append(newStates, determinizedGame)
		newReachProbs = append(newReachProbs, bs.reachProbs[i])
	}

	bs.states = newStates
	bs.reachProbs = newReachProbs
}

func (bs *BeliefState) determinizeDrawnCard(drawnCard cards.Card) {
	var newStates []*GameNode
	var newReachProbs []float32
	for i, game := range bs.states {
		state := game.GetState()
		drawPile := state.GetDrawPile()
		topCard := drawPile.NthCard(0)
		if topCard == cards.TBD {
			freeCards := getFreeCards(state)
			if !freeCards.Contains(drawnCard) {
				// This state could not possibly be valid because we drew a card
				// that was known not to be among the set of undetermined cards.
				continue
			}

			drawPile.SetNthCard(0, drawnCard)
			determinizedState := gamestate.NewShuffled(state, drawPile)
			determinizedGame := game.CloneWithState(determinizedState)
			newStates = append(newStates, determinizedGame)
			newReachProbs = append(newReachProbs, bs.reachProbs[i])
		} else if topCard == drawnCard {
			newStates = append(newStates, game)
			newReachProbs = append(newReachProbs, bs.reachProbs[i])
		} // else: state is incompatible with drawn card
	}

	bs.states = newStates
	bs.reachProbs = newReachProbs
}

func (bs *BeliefState) determinizeDrawnCardFromBottom(drawnCard cards.Card) {
	var newStates []*GameNode
	var newReachProbs []float32
	for i, game := range bs.states {
		state := game.GetState()
		drawPile := state.GetDrawPile()
		bottomCard := drawPile.NthCard(drawPile.Len() - 1)
		if bottomCard == cards.TBD {
			freeCards := getFreeCards(state)
			if !freeCards.Contains(drawnCard) {
				// This state could not possibly be valid because we drew a card
				// that was known not to be among the set of undetermined cards.
				continue
			}

			drawPile.SetNthCard(drawPile.Len()-1, drawnCard)
			determinizedState := gamestate.NewShuffled(state, drawPile)
			determinizedGame := game.CloneWithState(determinizedState)
			newStates = append(newStates, determinizedGame)
			newReachProbs = append(newReachProbs, bs.reachProbs[i])
		} else if bottomCard == drawnCard {
			newStates = append(newStates, game)
			newReachProbs = append(newReachProbs, bs.reachProbs[i])
		} // else: state is incompatible with drawn card
	}

	if len(newStates) == 0 {
		for i, game := range bs.states {
			glog.Errorf("Candidates state %d: %s", i, game)
			state := game.GetState()
			glog.Errorf("=> Draw pile: %s", state.GetDrawPile())
			glog.Errorf("=> P0 hand: %s", state.GetPlayerHand(gamestate.Player0))
			glog.Errorf("=> P1 hand: %s", state.GetPlayerHand(gamestate.Player1))
		}
		glog.Errorf("Looking to draw from the bottom a: %s", drawnCard)
		panic(fmt.Errorf("Belief state is empty!"))
	}

	bs.states = newStates
	bs.reachProbs = newReachProbs
}

func (bs *BeliefState) determinizeTopKCards(k int) {
	var newStates []*GameNode
	var newReachProbs []float32
	for i, game := range bs.states {
		state := game.GetState()
		determinizedDrawPiles := enumerateDrawPileDeterminizations(state, k)
		total := sumValues(determinizedDrawPiles)
		for determinizedDrawPile, freq := range determinizedDrawPiles {
			determinizedState := gamestate.NewShuffled(state, determinizedDrawPile)
			determinizedGame := game.CloneWithState(determinizedState)
			newStates = append(newStates, determinizedGame)
			chanceP := float32(freq) / float32(total)
			newReachProbs = append(newReachProbs, chanceP*bs.reachProbs[i])
		}
	}

	bs.states = newStates
	bs.reachProbs = newReachProbs
}

func (bs *BeliefState) determinizeForDrawFromTheBottom() {
	var newStates []*GameNode
	var newReachProbs []float32
	for i, game := range bs.states {
		// Determinize the bottom card so that DrawFromTheBottom is fully specified.
		determinizedState := game.GetState()
		drawPile := determinizedState.GetDrawPile()
		bottomCard := drawPile.NthCard(drawPile.Len() - 1)
		if bottomCard == cards.TBD {
			freeCards := getFreeCards(determinizedState)
			nFreeCards := freeCards.Len()
			freeCards.Iter(func(card cards.Card, count uint8) {
				drawPile.SetNthCard(drawPile.Len()-1, card)
				state := gamestate.NewShuffled(determinizedState, drawPile)
				determinizedGame := game.CloneWithState(state)
				newStates = append(newStates, determinizedGame)
				chanceP := float32(count) / float32(nFreeCards)
				newReachProbs = append(newReachProbs, chanceP*bs.reachProbs[i])
			})
		} else {
			// Bottom card has already been determinized.
			// No additional determinization expansion is necessary.
			newStates = append(newStates, game)
			newReachProbs = append(newReachProbs, bs.reachProbs[i])
		}
	}

	bs.states = newStates
	bs.reachProbs = newReachProbs
}

func clearDrawPileKnowledge(state gamestate.GameState) gamestate.GameState {
	drawPile := state.GetDrawPile()
	shuffledDrawPile := cards.NewStack()
	for i := 0; i < drawPile.Len(); i++ {
		shuffledDrawPile.SetNthCard(i, cards.TBD)
	}

	return gamestate.NewShuffled(state, shuffledDrawPile)
}

func (bs *BeliefState) SampleDeterminization() *GameNode {
	// First sample one of our belief states according to the reach probabilities.
	selected := sampleOne(bs.reachProbs)
	game := bs.states[selected]
	// Now sample a full determinization of this state uniformly, since all
	// unresolved determinizations are uniformly probable.
	determinizedState := sampleDeterminizedState(game.GetState())
	return game.CloneWithState(determinizedState)
}

func sampleDeterminizedState(state gamestate.GameState) gamestate.GameState {
	freeCards := getFreeCards(state)
	freeCardsSlice := freeCards.AsSlice()
	rand.Shuffle(len(freeCardsSlice), func(i, j int) {
		freeCardsSlice[i], freeCardsSlice[j] = freeCardsSlice[j], freeCardsSlice[i]
	})

	drawPile := state.GetDrawPile()
	for i := 0; i < drawPile.Len(); i++ {
		nthCard := drawPile.NthCard(i)
		if nthCard != cards.TBD {
			continue
		}

		nextCard := freeCardsSlice[0]
		drawPile.SetNthCard(i, nextCard)
		freeCardsSlice = freeCardsSlice[1:]
	}

	if len(freeCardsSlice) > 0 {
		panic(fmt.Errorf("Still have %d free cards remaining after determinization: %v", len(freeCardsSlice), freeCardsSlice))
	}

	return gamestate.NewShuffled(state, drawPile)
}

func getFreeCards(state gamestate.GameState) cards.Set {
	drawPile := state.GetDrawPile()
	p0Hand := state.GetPlayerHand(gamestate.Player0)
	p1Hand := state.GetPlayerHand(gamestate.Player1)
	h := state.GetHistory()

	freeCards := cards.CoreDeck
	freeCards.Add(cards.Defuse)
	freeCards.Add(cards.Defuse)
	freeCards.Add(cards.Defuse)
	freeCards.Add(cards.ExplodingKitten)

	// Remove all cards which are known to exist in either player's hand, a known position in the draw
	// pile, or have already been played.
	freeCards.RemoveAll(p0Hand)
	freeCards.RemoveAll(p1Hand)
	for i := 0; i < drawPile.Len(); i++ {
		nthCard := drawPile.NthCard(i)
		if nthCard != cards.Unknown && nthCard != cards.TBD {
			freeCards.Remove(nthCard)
		}
	}
	for i := 0; i < h.Len(); i++ {
		action := h.Get(i)
		if action.Type == gamestate.PlayCard {
			freeCards.Remove(action.Card)
		}
	}

	return freeCards
}

func enumerateDrawPileDeterminizations(state gamestate.GameState, n int) map[cards.Stack]int {
	drawPile := state.GetDrawPile()
	freeCards := getFreeCards(state)
	result := make(map[cards.Stack]int)
	enumerateDrawPilesHelper(freeCards, drawPile, n, 1, func(determinizedDrawPile cards.Stack, freq int) {
		result[determinizedDrawPile] += freq
	})

	return result
}

func enumerateDrawPilesHelper(deck cards.Set, result cards.Stack, n int, freq int, cb func(shuffle cards.Stack, freq int)) {
	if n == 0 { // All cards have been used, complete shuffle.
		cb(result, freq)
		return
	}

	nthCard := result.NthCard(n - 1)
	if nthCard == cards.TBD {
		deck.Iter(func(card cards.Card, count uint8) {
			// Take one of card from deck and append to result.
			remaining := deck
			remaining.Remove(card)
			newResult := result
			newResult.SetNthCard(n-1, card)

			// Recurse with remaining deck and new result.
			enumerateDrawPilesHelper(remaining, newResult, n-1, int(count)*freq, cb)
		})
	} else {
		// Nth card in the draw pile is already determined.
		enumerateDrawPilesHelper(deck, result, n-1, freq, cb)
	}
}

func uniformDistribution(n int) []float32 {
	result := make([]float32, n)
	for i := range result {
		result[i] = 1.0 / float32(n)
	}
	return result
}

func sumValues(m map[cards.Stack]int) int {
	total := 0
	for _, v := range m {
		total += v
	}
	return total
}

func sampleOne(vs []float32) int {
	total := sum(vs)
	x := total * rand.Float32()
	var cumSum float32
	for i, v := range vs {
		cumSum += v
		if cumSum > x {
			return i
		}
	}

	return len(vs) - 1
}

func sum(vs []float32) float32 {
	var total float32
	for _, v := range vs {
		total += v
	}
	return total
}
