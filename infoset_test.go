package alphacats

import (
	"reflect"
	"testing"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

func TestMarshalInfoset(t *testing.T) {
	isWithAvailableActions := InfoSetWithAvailableActions{
		InfoSet: gamestate.InfoSet{
			History: gamestate.NewHistoryFromActions([]gamestate.Action{
				{Player: gamestate.Player0, Type: gamestate.DrawCard},
				{Player: gamestate.Player0, Type: gamestate.DrawCard},
				{Player: gamestate.Player1, Type: gamestate.PlayCard, Card: cards.Shuffle},
				{Player: gamestate.Player1, Type: gamestate.InsertExplodingKitten},
			}),
			Hand: cards.NewSetFromCards([]cards.Card{cards.Cat, cards.Defuse, cards.Skip}),
		},
		AvailableActions: []gamestate.Action{
			{Player: gamestate.Player0, Type: gamestate.DrawCard},
			{Player: gamestate.Player1, Type: gamestate.PlayCard, Card: cards.Shuffle},
			{Player: gamestate.Player1, Type: gamestate.InsertExplodingKitten},
		},
	}

	buf, err := isWithAvailableActions.MarshalBinary()
	if err != nil {
		t.Error(err)
	}

	var reloaded InfoSetWithAvailableActions
	if err := reloaded.UnmarshalBinary(buf); err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(isWithAvailableActions, reloaded) {
		t.Errorf("expected: %v, got: %v", isWithAvailableActions, reloaded)
	}

	abstracted := newAbstractedInfoSet(isWithAvailableActions.InfoSet, isWithAvailableActions.AvailableActions)
	buf, err = abstracted.MarshalBinary()
	if err != nil {
		t.Error(err)
	}

	var reloadedAbstracted AbstractedInfoSet
	if err := reloadedAbstracted.UnmarshalBinary(buf); err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(*abstracted, reloadedAbstracted) {
		t.Errorf("expected: %v, got: %v", abstracted, reloadedAbstracted)
	}
}
