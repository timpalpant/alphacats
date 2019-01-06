package cards

import (
	"testing"
)

func TestNewSet(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSet(testCards)
	expected := map[Card]uint8{
		Unknown:      2,
		Skip:         1,
		Shuffle:      1,
		SeeTheFuture: 2,
	}

	for card, count := range expected {
		if set.CountOf(card) != count {
			t.Errorf("card set has %d of %v, expected %d", set.CountOf(card), card, count)
		}
	}
}

func TestLen(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSet(testCards)
	if set.Len() != 6 {
		t.Errorf("card set has len %d, expected %d", set.Len(), 6)
	}
}

func TestDistinct(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSet(testCards)
	expected := []Card{Unknown, Skip, Shuffle, SeeTheFuture}
	if !setEqual(set.Distinct(), expected) {
		t.Errorf("got unexpected set of distinct cards: %v", set.Distinct())
	}
}

func TestAsSlice(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSet(testCards)
	if !setEqual(set.AsSlice(), testCards) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}
}

func TestAdd(t *testing.T) {
	set := NewSet(nil)
	set.Add(Skip)
	if !setEqual(set.AsSlice(), []Card{Skip}) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}

	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set = NewSet(testCards)
	set.Add(Skip)
	if set.CountOf(Skip) != 2 {
		t.Error("failed to add Skip card")
	}

	expected := append(testCards, Skip)
	if !setEqual(set.AsSlice(), expected) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}
}

func TestRemove(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSet(testCards)
	set.Remove(Skip)
	if set.CountOf(Skip) != 0 {
		t.Error("failed to remove Skip card")
	}

	set.Remove(Unknown)
	if set.CountOf(Unknown) != 1 {
		t.Error("failed to remove Unknown card")
	}

	expected := []Card{Unknown, Shuffle, SeeTheFuture, SeeTheFuture}
	if !setEqual(set.AsSlice(), expected) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}
}

func TestRemove_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic when removing non-existent card")
		}
	}()

	set := NewSet([]Card{Skip})
	set.Remove(Shuffle)
}

func TestAddAll(t *testing.T) {
	set1 := NewSet([]Card{Skip})
	set2 := NewSet([]Card{Unknown, Unknown, Shuffle})
	set1.AddAll(set2)
	expected := []Card{Skip, Unknown, Unknown, Shuffle}
	if !setEqual(set1.AsSlice(), expected) {
		t.Errorf("got unexpected slice of cards: %v", set1)
	}
}

func TestRemoveAll(t *testing.T) {
	set1 := NewSet([]Card{Unknown, Unknown, Skip, Shuffle})
	set2 := NewSet([]Card{Unknown, Skip})
	set1.RemoveAll(set2)
	expected := []Card{Unknown, Shuffle}
	if !setEqual(set1.AsSlice(), expected) {
		t.Errorf("got unexpected slice of cards: %v", set1)
	}
}

func TestRemoveAll_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic when removing non-existent card")
		}
	}()

	set := NewSet([]Card{Skip})
	set2 := NewSet([]Card{Unknown, Skip})
	set.RemoveAll(set2)
}

func setEqual(s1, s2 []Card) bool {
	if len(s1) != len(s2) {
		return false
	}

	m1 := make(map[Card]int, len(s1))
	for _, card := range s1 {
		m1[card]++
	}

	for _, card := range s2 {
		m1[card]--
	}

	for _, count := range m1 {
		if count != 0 {
			return false
		}
	}

	return true
}
