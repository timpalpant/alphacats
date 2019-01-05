package alphacats

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
	t.Fail()
}

func TestInsertCard(t *testing.T) {
	t.Fail()
}
