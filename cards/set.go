package cards

import (
	"fmt"
	"strings"
)

// Set represents an unordered set of cards.
// Set[Card] is the number of that Card in the set.
type Set [Cat + 1]uint8

// NewSet creates a new et from the given slice of Cards.
func NewSet(cards []Card) Set {
	result := Set{}
	for _, card := range cards {
		result[card]++
	}

	return result
}

// CountOf gets the number of the given type of Card in the Set.
func (s Set) CountOf(card Card) uint8 {
	return s[card]
}

// Len gets the total number of Cards in the Set.
func (s Set) Len() int {
	result := 0
	for _, n := range s {
		result += int(n)
	}

	return result
}

// Distinct gets a slice of the distinct Cards in the Set.
func (s Set) Distinct() []Card {
	var result []Card
	for card, count := range s {
		if count > 0 {
			result = append(result, Card(card))
		}
	}

	return result
}

// AsSlice returns a slice of Cards with the given number of each
// Card as found in this Set.
func (s Set) AsSlice() []Card {
	var result []Card
	for card, count := range s {
		for i := uint8(0); i < count; i++ {
			result = append(result, Card(card))
		}
	}

	return result
}

// Add includes one of the given Card in the Set.
func (s *Set) Add(card Card) {
	(*s)[card]++
}

// Remove removes one of the given Card from the Set.
// Remove panics if the card is not present in the Set.
func (s *Set) Remove(card Card) {
	if (*s)[card] == 0 {
		panic(fmt.Errorf("card %v not in set", card))
	}

	(*s)[card]--
}

// AddAll adds the given cards to the Set.
func (s *Set) AddAll(cards Set) {
	result := (*s)
	for card, count := range cards {
		result[card] += count
	}

	*s = result
}

// RemoveAll removes the given cards from the set.
// RemoveAll panics if the cards are not present to be removed.
func (s *Set) RemoveAll(cards Set) {
	result := (*s)
	for card, count := range cards {
		if result[card] < count {
			panic(fmt.Errorf("cannot remove %d %v cards from set with only %d",
				count, card, result[card]))
		}

		result[card] -= count
	}

	*s = result
}

// String implements Stringer.
func (s Set) String() string {
	result := make([]string, 0)
	for card, count := range s {
		if count == 0 {
			continue
		}

		cardCount := fmt.Sprintf("%d %v", count, Card(card))
		result = append(result, cardCount)
	}

	return "{CardSet: " + strings.Join(result, ", ") + "}"
}
