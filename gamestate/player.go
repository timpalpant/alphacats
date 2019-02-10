package gamestate

// Player represents the identity of a player in the game.
type Player uint8

const (
	Player0 Player = iota
	Player1
)

var playerStr = [...]string{
	"Player0",
	"Player1",
}

func (p Player) String() string {
	return playerStr[p]
}
