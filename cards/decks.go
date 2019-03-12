package cards

// Cards in the 2-player core deck in the iOS app.
// Does not include the ExplodingCat, or the Defuse
var CoreDeck = NewSetFromCards([]Card{
	Skip, Skip, Skip, Skip, Skip,
	Slap1x, Slap1x, Slap1x,
	Slap2x,
	SeeTheFuture, SeeTheFuture, SeeTheFuture,
	Shuffle, Shuffle,
	DrawFromTheBottom, DrawFromTheBottom,
	Cat, Cat, Cat,
})

// Smaller deck for testing.
var TestDeck = NewSetFromCards([]Card{
	SeeTheFuture, Slap1x, Slap2x,
	Skip, DrawFromTheBottom, Cat,
})
