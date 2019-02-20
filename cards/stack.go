package cards

import (
	"fmt"
	"math/bits"
	"strings"
)

// Minimum number of bits required to store the identity of a card.
var (
	bitsPerCard = uint(bits.Len(uint(Cat)))
	topCardMask = Stack(1<<bitsPerCard) - 1
	maxCapacity = int(64 / bitsPerCard)
)

// Stack represents an ordered pile of cards, some of which may be unknown.
//
// The pile is encoded as a hexadecimal integer, where each digit (4 bits)
// corresponds to the identity of a Card.
// The top card in the pile is always the lowest order digit.
//
// NOTE: This implementation relies on the fact that there are at most
// 13 cards in the draw pile, and 10 distinct cards. Therefore, we can
// represent the identity of each card using 4 bits, and the identities
// of all 13 cards in the pile in 13 * 4 = 52 bits, or a single uint64.
// Cards that are Unknown are set to zero.
type Stack uint64

func NewStack() Stack {
	return Stack(0)
}

// NewStack creates a new Stack from the given slice of Cards.
func NewStackFromCards(cards []Card) Stack {
	result := Stack(0)
	for i, card := range cards {
		result.SetNthCard(i, card)
	}
	return result
}

func (s Stack) IsEmpty() bool {
	return s == 0
}

// Len is the number of cards in the Stack.
// Note that Unknown cards on the ends of the stack are not counted.
//
// Len is O(n) for the number of Cards in the Stack.
func (s Stack) Len() int {
	n := 0
	s.Iter(func(card Card) { n++ })
	return n
}

// SetNthCard returns a new CardPile with the identity of the Nth card
// in the stack set to card.
func (s *Stack) SetNthCard(n int, card Card) {
	shift := uint(n) * bitsPerCard
	// Zero out the current card in that position.
	*s &= ^Stack(topCardMask << shift)
	// Set the new card in that position
	*s |= Stack(uint64(card) << shift)
}

// NthCard returns the identity of the card in the Nth position of the stack.
// The Card may be Unknown.
func (s Stack) NthCard(n int) Card {
	shift := uint(n) * bitsPerCard
	return Card((s >> shift) & topCardMask)
}

// RemoveCard removes the Card in the Nth position.
func (s *Stack) RemoveCard(n int) {
	nBitsToKeep := uint(n) * bitsPerCard
	keepMask := Stack(1<<nBitsToKeep) - 1
	unchanged := (*s) & keepMask
	// Shift remaining cards one (but skip the nth card to remove it).
	toShift := (*s) &^ (keepMask << bitsPerCard)
	*s = unchanged + (toShift >> bitsPerCard)
}

// InsertCard places the given card inserted in the Nth position.
func (s *Stack) InsertCard(card Card, n int) {
	nBitsToKeep := uint(n) * bitsPerCard
	keepMask := Stack(1<<nBitsToKeep) - 1
	unchanged := (*s) & keepMask
	toShift := (*s) &^ keepMask
	// Shift remaining cards one to make room for the card we are inserting.
	*s = unchanged + (toShift << bitsPerCard)
	(*s).SetNthCard(n, card)
}

func (s Stack) Iter(cb func(card Card)) {
	for !s.IsEmpty() {
		card := Card(s & topCardMask)
		cb(card)
		s >>= bitsPerCard
	}
}

func (s Stack) ToSet() Set {
	set := NewSet()
	s.Iter(func(card Card) { set.Add(card) })
	return set
}

// String implements Stringer.
func (s Stack) String() string {
	cards := make([]string, 0)
	s.Iter(func(card Card) {
		cards = append(cards, fmt.Sprintf("%v", card))
	})

	return "[" + strings.Join(cards, ", ") + "]"
}
