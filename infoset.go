package alphacats

import (
	"encoding/binary"
	"encoding/gob"

	"github.com/timpalpant/alphacats/gamestate"
)

type InfoSetWithAvailableActions struct {
	gamestate.InfoSet
	AvailableActions []gamestate.Action
}

func (is InfoSetWithAvailableActions) MarshalBinary() ([]byte, error) {
	bufSize := is.InfoSet.MarshalBinarySize() + len(is.AvailableActions) + 2
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
	nAvailableActionBytes := len(buf) - nInfoSetBytes
	var nBuf [2]byte
	binary.LittleEndian.PutUint16(nBuf[:], uint16(nAvailableActionBytes))
	buf = append(buf, nBuf[:]...)

	return buf, nil
}

func (is *InfoSetWithAvailableActions) UnmarshalBinary(buf []byte) error {
	nAvailableActionBytes := int(binary.LittleEndian.Uint16(buf[len(buf)-2:]))
	buf = buf[:len(buf)-2]

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

func init() {
	gob.Register(&InfoSetWithAvailableActions{})
}
