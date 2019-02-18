package cards

import (
	"fmt"
	"strings"
)

const (
	bitsPerCardCount uint = 6
	maxCountPerType       = (1 << bitsPerCardCount) - 1
	mask                  = Set(1<<bitsPerCardCount) - 1
)

// Set represents an unordered set of cards.
// Set[Card] is the number of that Card in the set.
//
// The maximum value for a single type of Card is 63.
// Therefore the counts for all Cards can fit in a single uint64:
// 6 bits per Card x 10 types of Cards = 60 bits.
type Set uint64

func NewSet() Set {
	return Set(0)
}

// NewSetFromCards creates a new Set from the given slice of Cards.
func NewSetFromCards(cards []Card) Set {
	result := Set(0)
	for _, card := range cards {
		result.Add(card)
	}

	return result
}

// IsEmpty returns whether this Set contains any Cards.
func (s Set) IsEmpty() bool {
	return s == 0
}

// CountOf gets the number of the given type of Card in the Set.
func (s Set) CountOf(card Card) uint8 {
	shift := uint(card) * bitsPerCardCount
	return uint8((s >> shift) & mask)
}

// Contains returns whether the Set contains at least one of the given type of Card.
func (s Set) Contains(card Card) bool {
	return s.CountOf(card) > 0
}

// Counts returns a map of the number of each type of card in this Set.
func (s Set) Counts() map[Card]uint8 {
	result := make(map[Card]uint8)
	s.Iter(func(card Card, count uint8) {
		result[card] = count
	})
	return result
}

func (s Set) Iter(cb func(card Card, count uint8)) {
	for card := Card(0); s > 0; card++ {
		count := uint8(s & mask)
		if count > 0 {
			cb(card, count)
		}
		s >>= bitsPerCardCount
	}
}

// Len gets the total number of Cards in the Set.
func (s Set) Len() int {
	n := 0
	s.Iter(func(card Card, count uint8) {
		n += int(count)
	})
	return n
}

// Distinct gets a slice of the distinct Cards in the Set.
func (s Set) Distinct() []Card {
	var result []Card
	s.Iter(func(card Card, count uint8) {
		result = append(result, card)
	})
	return result
}

// AsSlice returns a slice of Cards with the given number of each
// Card as found in this Set.
func (s Set) AsSlice() []Card {
	var result []Card
	s.Iter(func(card Card, count uint8) {
		for i := uint8(0); i < count; i++ {
			result = append(result, card)
		}
	})

	return result
}

// Add includes one of the given Card in the Set.
func (s *Set) Add(card Card) {
	s.AddN(card, 1)
}

func (s *Set) AddN(card Card, n int) {
	shift := uint(card) * bitsPerCardCount
	*s += Set(n << shift)
}

// Remove removes one of the given Card from the Set.
// Remove panics if the card is not present in the Set.
func (s *Set) Remove(card Card) {
	s.RemoveN(card, 1)
}

func (s *Set) RemoveN(card Card, n int) {
	if int(s.CountOf(card)) < n {
		panic(fmt.Errorf("card %v not in set", card))
	}

	shift := uint(card) * bitsPerCardCount
	*s -= Set(n << shift)
}

// AddAll adds the given cards to the Set.
func (s *Set) AddAll(cards Set) {
	*s += cards
}

// RemoveAll removes the given cards from the set.
// RemoveAll panics if the cards are not present to be removed.
func (s *Set) RemoveAll(cards Set) {
	for card := Card(0); card <= Cat; card++ {
		if s.CountOf(card) < cards.CountOf(card) {
			panic(fmt.Errorf("cannot remove %d %v cards from set with only %d",
				cards.CountOf(card), card, s.CountOf(card)))
		}
	}

	*s -= cards
}

// String implements Stringer.
func (s Set) String() string {
	result := make([]string, 0)
	s.Iter(func(card Card, count uint8) {
		cardCount := fmt.Sprintf("%d %v", count, card)
		result = append(result, cardCount)
	})

	return "{" + strings.Join(result, ", ") + "}"
}
