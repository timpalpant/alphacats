package alphacats

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

type InfoSetWithAvailableActions struct {
	gamestate.InfoSet
	AvailableActions []gamestate.Action
}

func (is *InfoSetWithAvailableActions) MarshalBinary() ([]byte, error) {
	bufSize := is.InfoSet.MarshalBinarySize() + len(is.AvailableActions) + 1
	for _, action := range is.AvailableActions {
		if action.HasPrivateInfo() {
			bufSize += 2
		}
	}

	buf := make([]byte, 0, bufSize)
	buf, err := is.InfoSet.MarshalTo(buf)
	if err != nil {
		return nil, err
	}

	// Append available actions.
	nInfoSetBytes := len(buf)
	for _, action := range is.AvailableActions {
		packed := gamestate.EncodeAction(action)
		buf = append(buf, packed[0])

		// Actions are "varint" encoded: we only copy the private bits
		// if they are non-zero, which is indicated by the last bit of
		// the first byte.
		if action.HasPrivateInfo() {
			buf = append(buf, packed[1], packed[2])
		}
	}

	// Append number of available actions bytes so we can split off when unmarshaling.
	nAvailableActionBytes := uint8(len(buf) - nInfoSetBytes)
	buf = append(buf, nAvailableActionBytes)

	return buf, nil
}

func (is *InfoSetWithAvailableActions) UnmarshalBinary(buf []byte) error {
	nAvailableActionBytes := int(uint8(buf[len(buf)-1]))
	buf = buf[:len(buf)-1]

	actionsBuf := buf[len(buf)-nAvailableActionBytes:]
	isBuf := buf[:len(buf)-nAvailableActionBytes]
	if err := is.InfoSet.UnmarshalBinary(isBuf); err != nil {
		return err
	}

	if len(is.AvailableActions) > 0 { // Clear
		is.AvailableActions = is.AvailableActions[:0]
	}

	for len(actionsBuf) > 0 {
		packed := gamestate.EncodedAction{}
		packed[0] = actionsBuf[0]
		actionsBuf = actionsBuf[1:]

		if packed.HasPrivateInfo() {
			packed[1] = actionsBuf[0]
			packed[2] = actionsBuf[1]
			actionsBuf = actionsBuf[2:]
		}

		is.AvailableActions = append(is.AvailableActions, packed.Decode())
	}

	return nil
}

// AbstractedInfoSet abstracts away private history.
// The main difference in this abstraction is that the exact ordering in which
// private cards were received in the history is neglected.
// A second difference is that cards known to be in the draw pile (but not known where)
// are forgotten. This can happen if a SeeTheFuture card is played followed by
// a shuffle.
type AbstractedInfoSet struct {
	Player           gamestate.Player
	PublicHistory    gamestate.History
	Hand             cards.Set
	P0PlayedCards    cards.Set
	P1PlayedCards    cards.Set
	DrawPile         cards.Stack
	AvailableActions []gamestate.Action
}

func (a AbstractedInfoSet) String() string {
	return fmt.Sprintf("%s. Hand: %s, Draw pile: %s. Public history: %s. P0 played: %s, P1 played: %s. Available actions: %v",
		a.Player, a.Hand, a.DrawPile, a.PublicHistory, a.P0PlayedCards, a.P1PlayedCards, a.AvailableActions)
}

func newAbstractedInfoSet(is gamestate.InfoSet, availableActions []gamestate.Action) *AbstractedInfoSet {
	var publicHistory gamestate.History
	p0PlayedCards := cards.NewSet()
	p1PlayedCards := cards.NewSet()
	drawPile := cards.NewStack()
	// TODO(palpant): This duplicates most of gamestate logic, but from the POV of a single player.
	for i := 0; i < 13; i++ {
		drawPile.SetNthCard(i, cards.TBD)
	}
	for i := 0; i < is.History.Len(); i++ {
		packed := is.History.GetPacked(i)
		publicHistory.AppendPacked(hidePrivateInfo(packed))

		action := packed.Decode()
		switch action.Type {
		case gamestate.PlayCard:
			if action.Player == gamestate.Player0 {
				p0PlayedCards.Add(action.Card)
			} else {
				p1PlayedCards.Add(action.Card)
			}

			switch action.Card {
			case cards.SeeTheFuture:
				for i, card := range action.CardsSeen {
					if card != cards.Unknown {
						drawPile.SetNthCard(i, card)
					}
				}
			case cards.DrawFromTheBottom:
				drawPile.RemoveCard(drawPile.Len() - 1)
			case cards.Shuffle:
				drawPile = clearDrawPile(drawPile)
			}
		case gamestate.InsertExplodingKitten:
			if action.Player == gamestate.Player0 {
				p0PlayedCards.Add(cards.Defuse)
			} else {
				p1PlayedCards.Add(cards.Defuse)
			}

			if action.PositionInDrawPile != 0 {
				drawPile.InsertCard(cards.ExplodingKitten, int(action.PositionInDrawPile-1))
			} else {
				// TODO(palpant): Inserting the kitten randomly need not totally obliterate
				// our knowledge of the draw pile, since once we observe the Kitten we know
				// which of the N random states we're now in.
				drawPile.InsertCard(cards.ExplodingKitten, 0)
				drawPile = clearDrawPile(drawPile)
			}
		case gamestate.DrawCard:
			drawPile.RemoveCard(0)
		}
	}

	return &AbstractedInfoSet{
		Player:           is.Player,
		PublicHistory:    publicHistory,
		Hand:             is.Hand,
		P0PlayedCards:    p0PlayedCards,
		P1PlayedCards:    p1PlayedCards,
		DrawPile:         drawPile,
		AvailableActions: availableActions,
	}
}

func clearDrawPile(drawPile cards.Stack) cards.Stack {
	for j := 0; j < drawPile.Len(); j++ {
		drawPile.SetNthCard(j, cards.TBD)
	}
	return drawPile
}

func hidePrivateInfo(a gamestate.EncodedAction) gamestate.EncodedAction {
	a[1] = 0
	a[2] = 0
	return a
}

// Key implements cfr.InfoSet.
func (is *AbstractedInfoSet) Key() string {
	buf, _ := is.MarshalBinary()
	return string(buf)
}

func (is *AbstractedInfoSet) MarshalBinary() ([]byte, error) {
	// Doing extra work to exactly size the buffer (and avoid any additional
	// allocations ends up being faster than letting it auto-size)
	historySize := is.PublicHistory.Len() + 1
	for i := 0; i < is.PublicHistory.Len(); i++ {
		packed := is.PublicHistory.GetPacked(i)
		if packed.HasPrivateInfo() {
			historySize += 2
		}
	}

	cardsSize := 4 * 8
	availableActionsSize := len(is.AvailableActions) + 1
	for _, action := range is.AvailableActions {
		if action.HasPrivateInfo() {
			availableActionsSize += 2
		}
	}

	bufSize := cardsSize + historySize + availableActionsSize
	buf := make([]byte, 0, bufSize)

	// First do sets of cards.
	var hBuf [8]byte
	binary.LittleEndian.PutUint64(hBuf[:], uint64(is.Hand))
	buf = append(buf, hBuf[:]...)
	binary.LittleEndian.PutUint64(hBuf[:], uint64(is.P0PlayedCards))
	buf = append(buf, hBuf[:]...)
	binary.LittleEndian.PutUint64(hBuf[:], uint64(is.P1PlayedCards))
	buf = append(buf, hBuf[:]...)
	// Then draw pile.
	binary.LittleEndian.PutUint64(hBuf[:], uint64(is.DrawPile))
	buf = append(buf, hBuf[:]...)

	// Then history, prefixed by length.
	buf = append(buf, uint8(is.PublicHistory.Len()))
	for i := 0; i < is.PublicHistory.Len(); i++ {
		action := is.PublicHistory.GetPacked(i)
		buf = append(buf, action[0])

		// Actions are "varint" encoded: we only copy the private bits
		// if they are non-zero, which is indicated by the last bit of
		// the first byte.
		if action.HasPrivateInfo() {
			buf = append(buf, action[1], action[2])
		}
	}

	// Then available actions, prefixed by length.
	buf = append(buf, uint8(len(is.AvailableActions)))
	for _, action := range is.AvailableActions {
		packed := gamestate.EncodeAction(action)
		buf = append(buf, packed[0])

		// Actions are "varint" encoded: we only copy the private bits
		// if they are non-zero, which is indicated by the last bit of
		// the first byte.
		if action.HasPrivateInfo() {
			buf = append(buf, packed[1], packed[2])
		}
	}

	return buf, nil
}

func (is *AbstractedInfoSet) UnmarshalBinary(buf []byte) error {
	is.PublicHistory.Clear()

	is.Hand = cards.Set(binary.LittleEndian.Uint64(buf))
	buf = buf[8:]
	is.P0PlayedCards = cards.Set(binary.LittleEndian.Uint64(buf))
	buf = buf[8:]
	is.P1PlayedCards = cards.Set(binary.LittleEndian.Uint64(buf))
	buf = buf[8:]
	is.DrawPile = cards.Stack(binary.LittleEndian.Uint64(buf))
	buf = buf[8:]

	nActions := uint8(buf[0])
	buf = buf[1:]
	for i := 0; i < int(nActions); i++ {
		var packed gamestate.EncodedAction
		packed[0] = buf[0]
		buf = buf[1:]

		if packed.HasPrivateInfo() {
			packed[1] = buf[0]
			packed[2] = buf[1]
			buf = buf[2:]
		}

		is.PublicHistory.AppendPacked(packed)
	}

	if len(is.AvailableActions) > 0 { // Clear
		is.AvailableActions = is.AvailableActions[:0]
	}

	nActions = uint8(buf[0])
	buf = buf[1:]
	for len(buf) > 0 {
		packed := gamestate.EncodedAction{}
		packed[0] = buf[0]
		buf = buf[1:]

		if packed.HasPrivateInfo() {
			packed[1] = buf[0]
			packed[2] = buf[1]
			buf = buf[2:]
		}

		is.AvailableActions = append(is.AvailableActions, packed.Decode())
	}

	if len(is.AvailableActions) != int(nActions) {
		panic(fmt.Errorf("expected %d available actions, got %d",
			nActions, len(is.AvailableActions)))
	}

	return nil
}

func init() {
	gob.Register(&InfoSetWithAvailableActions{})
	gob.Register(&AbstractedInfoSet{})
}
