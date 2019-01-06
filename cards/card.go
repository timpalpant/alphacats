package cards

// Card represents one card from the Exploding Kittens game deck.
type Card int

const (
	Unknown Card = iota
	ExplodingCat
	Defuse
	Skip
	Slap1x
	Slap2x
	SeeTheFuture
	Shuffle
	DrawFromTheBottom
	Cat // Must be last since it is used to size CardPile and CardSet
)

var cardStr = [...]string{
	"Unknown",
	"ExplodingCat",
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
