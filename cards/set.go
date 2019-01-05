package alphacats

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

// Add returns a new Set with the given card added to it.
func (s Set) Add(card Card) Set {
	result := s
	result[card]++
	return result
}

// Remove returns a new Set with the given card removed.
// Remove panics if the card is not present in the Set.
func (s Set) Remove(card Card) Set {
	result := s
	if result[card] == 0 {
		panic(fmt.Errorf("card %v not in set", card))
	}

	result[card]--
	return result
}

// AddAll returns a new Set with the given cards added to it.
func (s Set) AddAll(cards Set) Set {
	result := s
	for card, count := range cards {
		result[card] += count
	}

	return result
}

// RemoveAll returns a new Set with the given cards removed from it.
// RemoveAll panics if the cards are not present to be removed.
func (s Set) RemoveAll(cards Set) Set {
	result := s
	for card, count := range cards {
		if result[card] < count {
			panic(fmt.Errorf("cannot remove %d %v cards from set with only %d",
				count, card, result[card]))
		}

		result[card] -= count
	}

	return result
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
