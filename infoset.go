package alphacats

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"unsafe"

	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

// The max number of actions that we can remember.
const MaxMemory = 4

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

type AbstractedInfoSet struct {
	RecentHistory gamestate.History
	Hand          cards.Set
	P0PlayedCards cards.Set
	P1PlayedCards cards.Set
	// TODO(palpant): Known info about other player's hand due to given cards.
	AvailableActions []gamestate.Action
}

func newAbstractedInfoSet(is gamestate.InfoSet, availableActions []gamestate.Action) *AbstractedInfoSet {
	var recentHistory gamestate.History
	p0PlayedCards := cards.NewSet()
	p1PlayedCards := cards.NewSet()
	for i := 0; i < is.History.Len(); i++ {
		packed := is.History.GetPacked(i)
		if i < MaxMemory {
			recentHistory.AppendPacked(packed)
		}

		action := packed.Decode()
		if action.Type == gamestate.PlayCard {
			if action.Player == gamestate.Player0 {
				p0PlayedCards.Add(action.Card)
			} else {
				p1PlayedCards.Add(action.Card)
			}
		}
	}

	return &AbstractedInfoSet{
		RecentHistory:    recentHistory,
		Hand:             is.Hand,
		P0PlayedCards:    p0PlayedCards,
		P1PlayedCards:    p1PlayedCards,
		AvailableActions: availableActions,
	}
}

// Key implements cfr.InfoSet.
func (is *AbstractedInfoSet) Key() string {
	buf, _ := is.MarshalBinary()
	return *(*string)(unsafe.Pointer(&buf))
}

func (is *AbstractedInfoSet) MarshalBinary() ([]byte, error) {
	// Doing extra work to exactly size the buffer (and avoid any additional
	// allocations ends up being faster than letting it auto-size)
	historySize := is.RecentHistory.Len() + 1
	for i := 0; i < is.RecentHistory.Len(); i++ {
		packed := is.RecentHistory.GetPacked(i)
		if packed.HasPrivateInfo() {
			historySize += 2
		}
	}

	cardsSize := 3 * 8
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

	// Then history, prefixed by length.
	buf = append(buf, uint8(is.RecentHistory.Len()))
	for i := 0; i < is.RecentHistory.Len(); i++ {
		action := is.RecentHistory.GetPacked(i)
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
	is.RecentHistory.Clear()

	is.Hand = cards.Set(binary.LittleEndian.Uint64(buf))
	buf = buf[8:]
	is.P0PlayedCards = cards.Set(binary.LittleEndian.Uint64(buf))
	buf = buf[8:]
	is.P1PlayedCards = cards.Set(binary.LittleEndian.Uint64(buf))
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

		is.RecentHistory.AppendPacked(packed)
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
