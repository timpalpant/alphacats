package alphacats

import (
	"fmt"
	"math/bits"
	"math/rand"
	"strings"
)

// Cards in the 2-player core deck in the iOS app.
// Does not include the ExplodingCat, or the Defuse cards initially
// dealt to each player.
var CoreDeck = CardSet{0, 0, 1, 5, 3, 1, 3, 2, 2, 3}

// Card represents one card from the Exploding Kittens game deck.
type Card int

const (
	Unknown Card = iota
	ExplodingCat
	Defuse
	Skip
	Slap1x
	Slap2x
	SeeTheFuture
	Shuffle
	DrawFromTheBottom
	Cat // Must be last since it is used to size CardPile and CardSet
)

var cardStr = [...]string{
	"Unknown",
	"ExplodingCat",
	"Defuse",
	"Skip",
	"Slap1x",
	"Slap2x",
	"SeeTheFuture",
	"Shuffle",
	"DrawFromTheBottom",
	"Cat",
}

// String implements Stringer.
func (c Card) String() string {
	return cardStr[c]
}

// Minimum number of bits required to store the identity of a card.
var (
	bitsPerCard = uint(bits.Len(uint(Cat)))
	topCardMask = CardPile(1<<bitsPerCard) - 1
)

// CardPile represents an ordered pile of known cards.
// The pile is encoded as a hexadecimal integer, where each digit (4 bits)
// corresponds to the identity of a Card.
// The top card in the pile is always the lowest order digit.
type CardPile uint64

func (cp CardPile) SetNthCard(n int, card Card) CardPile {
	result := cp
	result -= CardPile(cp.NthCard(n))
	shift := uint(n) * bitsPerCard
	result += CardPile(card << shift)
	return result
}

func (cp CardPile) NthCard(n int) Card {
	shift := uint(n) * bitsPerCard
	return Card((cp >> shift) & topCardMask)
}

// String implements Stringer.
func (cp CardPile) String() string {
	cards := make([]string, 0)
	for cp > 0 {
		c := Card(cp & topCardMask)
		cards = append(cards, fmt.Sprintf("%v", c))
		cp >>= bitsPerCard
	}

	return "[CardPile: " + strings.Join(cards, ", ") + "]"
}

// CardSet represents an unordered set of cards.
// CardSet[Card] is the number of that Card in the set.
type CardSet [Cat + 1]uint8

func NewCardSet(cards []Card) CardSet {
	result := CardSet{}
	for _, card := range cards {
		result[card]++
	}

	return result
}

func (cs CardSet) Count() int {
	result := 0
	for _, n := range cs {
		result += int(n)
	}

	return result
}

func (cs CardSet) AsSlice() []Card {
	var result []Card
	for card, count := range cs {
		for i := uint8(0); i < count; i++ {
			result = append(result, Card(card))
		}
	}

	return result
}

func (cs CardSet) DrawRandom(n int) CardSet {
	cards := cs.AsSlice()
	rand.Shuffle(len(cards), func(i, j int) {
		cards[i], cards[j] = cards[j], cards[i]
	})

	return NewCardSet(cards[:n])
}

func (cs CardSet) Remove(cards CardSet) CardSet {
	result := cs
	for card, count := range cards {
		result[card] -= count
	}

	return result
}

// String implements Stringer.
func (cs CardSet) String() string {
	result := make([]string, 0)
	for card, count := range cs {
		if count == 0 {
			continue
		}

		cardCount := fmt.Sprintf("%d %v", count, Card(card))
		result = append(result, cardCount)
	}

	return "{CardSet: " + strings.Join(result, ", ") + "}"
}
