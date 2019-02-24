package model

import (
	"encoding/binary"
	"io"

	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

// Samples are saved to the writer as a flat list of marshaled protos,
// prefixed by their size encoded as a Uvarint64:
//
//  [size] [Sample proto] [size] [Sample proto] ...
func saveTrainingData(samples []deepcfr.Sample, w io.Writer) error {
	var sizeBuf [binary.MaxVarintLen64]byte
	for _, sample := range samples {
		sampleProto := sampleToProto(sample)
		buf, err := sampleProto.Marshal()
		if err != nil {
			return err
		}

		n := binary.PutUvarint(sizeBuf[:], uint64(len(buf)))
		if _, err := w.Write(sizeBuf[:n]); err != nil {
			return err
		}

		if _, err := w.Write(buf); err != nil {
			return err
		}
	}

	return nil
}

func sampleToProto(sample deepcfr.Sample) *Sample {
	return &Sample{
		Iter:       uint64(sample.Iter),
		Infoset:    infosetToProto(sample.InfoSet),
		Advantages: sample.Advantages,
	}
}

func infosetToProto(infoSet cfr.InfoSet) *InfoSet {
	is := infoSet.(*gamestate.InfoSet)
	return &InfoSet{
		History: historyToProto(is.History),
		Hand:    cardsToProto(is.Hand),
	}
}

func historyToProto(h gamestate.History) []*Action {
	result := make([]*Action, h.Len())
	for i, action := range h.AsSlice() {
		result[i] = actionToProto(action)
	}

	return result
}

func actionToProto(action gamestate.Action) *Action {
	var cardsSeen []Card
	for _, card := range action.CardsSeen {
		if card != cards.Unknown {
			cardsSeen = append(cardsSeen, Card(card))
		}
	}

	return &Action{
		Player:             int32(action.Player),
		Type:               Action_Type(action.Type),
		Card:               Card(action.Card),
		PositionInDrawPile: int32(action.PositionInDrawPile),
		CardsSeen:          cardsSeen,
	}
}

func cardsToProto(hand cards.Set) []Card {
	result := make([]Card, hand.Len())
	for i, card := range hand.AsSlice() {
		result[i] = Card(card)
	}

	return result
}
