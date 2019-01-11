package cards

import (
	"testing"
)

func TestNewSetFromCards(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSetFromCards(testCards)
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
	set := NewSetFromCards(testCards)
	if set.Len() != 6 {
		t.Errorf("card set has len %d, expected %d", set.Len(), 6)
	}
}

func TestDistinct(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSetFromCards(testCards)
	expected := []Card{Unknown, Skip, Shuffle, SeeTheFuture}
	if !setEqual(set.Distinct(), expected) {
		t.Errorf("got unexpected set of distinct cards: %v", set.Distinct())
	}
}

func TestAsSlice(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSetFromCards(testCards)
	if !setEqual(set.AsSlice(), testCards) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}
}

func TestAdd(t *testing.T) {
	set := NewSetFromCards(nil)
	set.Add(Skip)
	if !setEqual(set.AsSlice(), []Card{Skip}) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}

	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set = NewSetFromCards(testCards)
	set.Add(Skip)
	if set.CountOf(Skip) != 2 {
		t.Error("failed to add Skip card")
	}

	expected := append(testCards, Skip)
	if !setEqual(set.AsSlice(), expected) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}
}

func TestAddN(t *testing.T) {
	set := NewSet()
	set.AddN(Skip, 3)
	if !setEqual(set.AsSlice(), []Card{Skip, Skip, Skip}) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}

	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set = NewSetFromCards(testCards)
	set.AddN(Skip, 2)
	if set.CountOf(Skip) != 3 {
		t.Error("failed to add Skip cards")
	}

	expected := append(testCards, Skip, Skip)
	if !setEqual(set.AsSlice(), expected) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}

}

func TestRemove(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSetFromCards(testCards)
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

	set := NewSetFromCards([]Card{Skip})
	set.Remove(Shuffle)
}

func TestRemoveN(t *testing.T) {
	testCards := []Card{Unknown, Unknown, Skip, Shuffle, SeeTheFuture, SeeTheFuture}
	set := NewSetFromCards(testCards)
	set.RemoveN(Unknown, 2)
	if set.CountOf(Unknown) != 0 {
		t.Error("failed to remove Unknown cards")
	}

	set.RemoveN(SeeTheFuture, 1)
	if set.CountOf(SeeTheFuture) != 1 {
		t.Error("failed to remove SeeTheFuture card")
	}

	expected := []Card{Skip, Shuffle, SeeTheFuture}
	if !setEqual(set.AsSlice(), expected) {
		t.Errorf("got unexpected slice of cards: %v", set)
	}

}

func TestRemoveN_Panic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic when removing non-existent card")
		}
	}()

	set := NewSetFromCards([]Card{Skip})
	set.RemoveN(Skip, 2)
}

func TestAddAll(t *testing.T) {
	set1 := NewSetFromCards([]Card{Skip})
	set2 := NewSetFromCards([]Card{Unknown, Unknown, Shuffle})
	set1.AddAll(set2)
	expected := []Card{Skip, Unknown, Unknown, Shuffle}
	if !setEqual(set1.AsSlice(), expected) {
		t.Errorf("got unexpected slice of cards: %v", set1)
	}
}

func TestRemoveAll(t *testing.T) {
	set1 := NewSetFromCards([]Card{Unknown, Unknown, Skip, Shuffle})
	set2 := NewSetFromCards([]Card{Unknown, Skip})
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

	set := NewSetFromCards([]Card{Skip})
	set2 := NewSetFromCards([]Card{Unknown, Skip})
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
