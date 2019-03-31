package alphacats

import (
	"encoding/gob"

	"github.com/timpalpant/alphacats/gamestate"
)

type InfoSetWithAvailableActions struct {
	*gamestate.InfoSet
	AvailableActions []gamestate.Action
}

func (is *InfoSetWithAvailableActions) MarshalBinary() ([]byte, error) {
	buf, err := is.InfoSet.MarshalBinary()
	if err != nil {
		return nil, err
	}

	// Append available actions.
	nBytes := 3*len(is.AvailableActions) + 1
	buf = append(buf, make([]byte, nBytes)...)
	aBuf := buf[len(buf)-nBytes:]
	for i, action := range is.AvailableActions {
		packed := gamestate.EncodeAction(action)
		aBuf[3*i] = packed[0]
		aBuf[3*i+1] = packed[1]
		aBuf[3*i+2] = packed[2]
	}

	// Append number of available actions so we can unmarshal.
	buf[len(buf)-1] = uint8(len(is.AvailableActions))
	return buf, nil
}

func (is *InfoSetWithAvailableActions) UnmarshalBinary(buf []byte) error {
	nActions := int(uint8(buf[len(buf)-1]))
	buf = buf[:len(buf)-1]

	actionsBuf := buf[len(buf)-3*nActions:]
	buf = buf[:len(buf)-3*nActions]
	is.InfoSet = &gamestate.InfoSet{}
	if err := is.InfoSet.UnmarshalBinary(buf); err != nil {
		return err
	}

	is.AvailableActions = make([]gamestate.Action, nActions)
	for i := range is.AvailableActions {
		packed := gamestate.EncodedAction{}
		packed[0] = actionsBuf[3*i]
		packed[1] = actionsBuf[3*i+1]
		packed[2] = actionsBuf[3*i+2]
		is.AvailableActions[i] = packed.Decode()
	}

	return nil
}

func init() {
	gob.Register(&InfoSetWithAvailableActions{})
}
