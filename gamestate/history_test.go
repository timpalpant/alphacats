package gamestate

import (
	"reflect"
	"testing"

	"github.com/timpalpant/alphacats/cards"
)

func TestEncodeDecode(t *testing.T) {
	for _, player := range []Player{Player0, Player1} {
		for _, actionType := range allActions {
			for card := cards.Card(0); card <= cards.Cat; card++ {
				action := Action{Player: player, Type: actionType, Card: card}
				packed := EncodeAction(action)
				decoded := packed.Decode()
				if !reflect.DeepEqual(action, decoded) {
					t.Errorf("input: %+v, output: %+v", action, decoded)
				}
			}
		}
	}
}

func TestHasPrivateInfo(t *testing.T) {
	action := Action{Player: 0, Type: DrawCard, Card: cards.Unknown}
	packed := EncodeAction(action)
	if packed.HasPrivateInfo() {
		t.Errorf("action does not have private info: %v", packed)
	}

	action.PositionInDrawPile = 1
	packed = EncodeAction(action)
	if !packed.HasPrivateInfo() {
		t.Errorf("action has private info: %v", packed)
	}

	action.PositionInDrawPile = 0
	action.CardsSeen[0] = cards.Skip
	packed = EncodeAction(action)
	if !packed.HasPrivateInfo() {
		t.Errorf("action has private info: %v", packed)
	}
}

func TestPackSequences(t *testing.T) {
	testCases := [][]Action{
		{
			{Player: Player0, Type: DrawCard},
			{Player: Player0, Type: DrawCard, CardsSeen: [3]cards.Card{
				cards.Slap1x, cards.Unknown, cards.Unknown,
			}},
			{Player: Player1, Type: PlayCard, Card: cards.Shuffle},
			{Player: Player1, Type: InsertExplodingCat, PositionInDrawPile: 13},
			{Player: Player1, Type: InsertExplodingCat, PositionInDrawPile: 11},
		},
		{
			{Player: Player0, Type: PlayCard, Card: cards.Skip},
			{Player: Player0, Type: PlayCard, Card: cards.SeeTheFuture, CardsSeen: [3]cards.Card{
				cards.Cat, cards.Skip, cards.SeeTheFuture,
			}},
			{Player: Player0, Type: PlayCard, Card: cards.Cat},
		},
	}

	for _, testCase := range testCases {
		h := newHistoryFromSlice(testCase)
		result := h.AsSlice()
		if !reflect.DeepEqual(result, testCase) {
			t.Errorf("input: %+v, output: %+v", testCase, result)
		}
	}
}

func newHistoryFromSlice(actions []Action) History {
	h := History{}
	for _, action := range actions {
		h.Append(action)
	}
	return h
}

func TestAppend(t *testing.T) {
	testSequence := []Action{
		{Player: Player0, Type: DrawCard},
		{Player: Player0, Type: DrawCard},
		{Player: Player1, Type: PlayCard, Card: cards.Shuffle},
	}

	h := newHistoryFromSlice(testSequence)
	action := Action{Player: Player1, Type: InsertExplodingCat}
	h.Append(action)
	result := h.AsSlice()
	if len(result) != len(testSequence)+1 {
		t.Errorf("expected %v items, got %v", len(testSequence)+1, len(result))
	}

	lastItem := result[len(result)-1]
	if !reflect.DeepEqual(lastItem, action) {
		t.Errorf("expected %v, got %v", action, lastItem)
	}
}

func TestMarshalInfoSet(t *testing.T) {
	testCases := []struct {
		history []Action
		hand    []cards.Card
	}{
		{
			history: []Action{
				{Player: Player0, Type: DrawCard},
				{Player: Player0, Type: DrawCard},
				{Player: Player1, Type: PlayCard, Card: cards.Shuffle},
				{Player: Player1, Type: InsertExplodingCat},
			},
			hand: []cards.Card{cards.Cat, cards.Defuse, cards.Skip},
		},
		{
			history: []Action{
				{Player: Player0, Type: PlayCard, Card: cards.Skip},
				{Player: Player0, Type: PlayCard, Card: cards.SeeTheFuture},
				{Player: Player0, Type: PlayCard, Card: cards.Cat},
			},
			hand: []cards.Card{cards.ExplodingCat, cards.Unknown},
		},
	}

	for _, tc := range testCases {
		history := EncodeActions(tc.history)
		hand := cards.NewSetFromCards(tc.hand)
		is := InfoSet{history, hand}

		buf, err := is.MarshalBinary()
		if err != nil {
			t.Error(err)
		}

		var unmarshaled InfoSet
		if err := unmarshaled.UnmarshalBinary(buf); err != nil {
			t.Error(err)
		}

		if !reflect.DeepEqual(is, unmarshaled) {
			t.Errorf("expected %v, got %v", is, unmarshaled)
		}
	}
}
