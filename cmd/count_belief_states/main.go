package main

import (
	"flag"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"

	"github.com/golang/glog"
)

func enumerateDeals(deck cards.Set, result cards.Set, n int, cb func(deal cards.Set)) {
	if n == 0 {
		cb(result)
		return
	}

	deck.Iter(func(card cards.Card, count uint8) {
		remaining := deck
		remaining.Remove(card)
		newResult := result
		newResult.Add(card)
		enumerateDeals(remaining, newResult, n-1, cb)
	})
}

func main() {
	flag.Parse()

	p0Deals := make(map[cards.Set]struct{})
	enumerateDeals(cards.CoreDeck, cards.NewSet(), 4, func(deal cards.Set) {
		p0Deals[deal] = struct{}{}
	})
	glog.Infof("%d distinct player0 deals", len(p0Deals))

	nDeals := 0
	totalInitialStates := 0
	maxBeliefStates := 0
	maxDeckShuffles := 0
	for p0Deal := range p0Deals {
		glog.Infof("P0 deal: %v", p0Deal)
		remaining := cards.CoreDeck
		remaining.RemoveAll(p0Deal)
		p1Deals := make(map[cards.Set]struct{})
		nBeliefStates := 0
		enumerateDeals(remaining, cards.NewSet(), 4, func(p1Deal cards.Set) {
			p1Deals[p1Deal] = struct{}{}
			drawPile := remaining
			drawPile.RemoveAll(p1Deal)
			drawPile.Add(cards.Defuse)
			drawPile.Add(cards.ExplodingKitten)
			nShuffles := alphacats.CountDistinctShuffles(drawPile)
			nDeals++
			nBeliefStates += nShuffles
			if nShuffles > maxDeckShuffles {
				maxDeckShuffles = nShuffles
			}
		})

		glog.Infof("=> %d distinct player1 deals", len(p1Deals))
		glog.Infof("=> %d belief states", nBeliefStates)
		totalInitialStates += nBeliefStates
		if nBeliefStates > maxBeliefStates {
			maxBeliefStates = nBeliefStates
		}
	}

	glog.Infof("%d initial deals", nDeals)
	glog.Infof("%d total initial states", totalInitialStates)
	glog.Infof("Max number of initial belief states: %d", maxBeliefStates)
	glog.Infof("Max number of deck shuffles: %d", maxDeckShuffles)

	glog.Info("Enumerating initial states")
	n := 0
	alphacats.EnumerateInitialDeals(cards.CoreDeck, 4, func(deal alphacats.Deal) {
		n++
		if n%100000000 == 0 {
			glog.Infof("Enumerated %d initial states", n)
		}
	})
	glog.Infof("Enumerated %d initial states", n)
}
