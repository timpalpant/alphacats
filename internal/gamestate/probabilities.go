package gamestate

import (
	"github.com/golang/glog"

	"github.com/timpalpant/alphacats/cards"
)

var (
	fixedCardProbabilities = make([]map[cards.Card]float64, cards.Cat+1)
	cardProbabilitiesCache = computeCardProbabilitiesMap()
)

func init() {
	for card := cards.Card(0); card <= cards.Cat; card++ {
		fixedCardProbabilities[card] = map[cards.Card]float64{card: 1.0}
	}
}

func computeCardProbabilitiesMap() map[cards.Set]map[cards.Card]float64 {
	glog.Info("Building card probabilities map")
	deck := cards.CoreDeck
	deck.Add(cards.ExplodingCat)
	glog.Info("Enumerating all possible subsets of deck")
	allSubsets := enumerateAllSubsets(deck, cards.NewSet(), cards.Card(0), nil)

	glog.Info("Converting to probabilities")
	result := make(map[cards.Set]map[cards.Card]float64, len(allSubsets))
	for _, subset := range allSubsets {
		result[subset] = toProbabilities(subset)
	}

	glog.Infof("Built map with %d items", len(result))
	return result
}

func enumerateAllSubsets(available, current cards.Set, card cards.Card, result []cards.Set) []cards.Set {
	if card > cards.Cat {
		return result
	}

	count := int(available.CountOf(card))
	for i := 0; i <= count; i++ {
		current.AddN(card, i)
		result = append(result, current)
		result = enumerateAllSubsets(available, current, card+1, result)
		current.RemoveN(card, i)
	}

	return result
}

func toProbabilities(candidates cards.Set) map[cards.Card]float64 {
	counts := candidates.Counts()
	result := make(map[cards.Card]float64, len(counts))
	nTotal := float64(candidates.Len())
	for card, count := range counts {
		p := float64(count) / nTotal
		result[card] = p
	}

	return result
}
