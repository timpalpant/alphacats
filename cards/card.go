package cards

// Card represents one card from the Exploding Kittens game deck.
type Card uint8

const (
	Unknown Card = iota
	ExplodingKitten
	Defuse
	Skip
	Slap1x
	Slap2x
	SeeTheFuture
	Shuffle
	DrawFromTheBottom
	Cat
)

var cardStr = [...]string{
	"Unknown",
	"ExplodingKitten",
	"Defuse",
	"Skip",
	"Slap1x",
	"Slap2x",
	"SeeTheFuture",
	"Shuffle",
	"DrawFromTheBottom",
	"Cat",
}

// String implements Stringer.
func (c Card) String() string {
	return cardStr[c]
}

// The number of distinct types of Cards.
const NumTypes = len(cardStr)
