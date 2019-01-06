package cards

import (
	"testing"
)

func TestNewStack(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	stack := NewStack(testCards)
	for i, card := range testCards {
		if stack.NthCard(i) != card {
			t.Errorf("card pile position %d has %v, expected %v", i, stack.NthCard(i), card)
		}
	}
}

func TestSetNthCard(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	stack := NewStack(testCards)
	stack.SetNthCard(1, Slap1x)
	testCards[1] = Slap1x
	for i, card := range testCards {
		if stack.NthCard(i) != card {
			t.Errorf("card pile position %d has %v, expected %v", i, stack.NthCard(i), card)
		}
	}

	if stack.NthCard(10) != Unknown {
		t.Errorf("card pile position %d has %v, expected %v", 10, stack.NthCard(10), Unknown)
	}
	stack.SetNthCard(10, Slap2x)
	if stack.NthCard(10) != Slap2x {
		t.Errorf("card pile position %d has %v, expected %v", 10, stack.NthCard(10), Slap2x)
	}

	// All other positions should remain the same.
	for i, card := range testCards {
		if stack.NthCard(i) != card {
			t.Errorf("card pile position %d has %v, expected %v", i, stack.NthCard(i), card)
		}
	}
}

func TestRemoveCard(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	stack := NewStack(testCards)
	stack.RemoveCard(0)
	testCards = testCards[1:]
	for i, card := range testCards {
		if stack.NthCard(i) != card {
			t.Errorf("card pile position %d has %v, expected %v", i, stack.NthCard(i), card)
			t.Logf("%v, expected %v", stack, testCards)
		}
	}

	stack.RemoveCard(2)
	testCards = append(testCards[:2], testCards[3:]...)
	for i, card := range testCards {
		if stack.NthCard(i) != card {
			t.Errorf("card pile position %d has %v, expected %v", i, stack.NthCard(i), card)
		}
	}
}

func TestInsertCard(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	stack := NewStack(testCards)
	stack.InsertCard(Slap1x, 0)
	testCards = append([]Card{Slap1x}, testCards...)
	for i, card := range testCards {
		if stack.NthCard(i) != card {
			t.Errorf("card pile position %d has %v, expected %v", i, stack.NthCard(i), card)
			t.Logf("%v, expected %v", stack, testCards)
		}
	}

	stack.InsertCard(Slap2x, 2)
	remainder := append([]Card{Slap2x}, testCards[2:]...)
	testCards = append(testCards[:2], remainder...)
	for i, card := range testCards {
		if stack.NthCard(i) != card {
			t.Errorf("card pile position %d has %v, expected %v", i, stack.NthCard(i), card)
			t.Logf("%v, expected %v", stack, testCards)
		}
	}
}
