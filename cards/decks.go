package cards

// Cards in the 2-player core deck in the iOS app.
// Does not include the ExplodingCat, or the Defuse cards initially
// dealt to each player.
var CoreDeck = NewSetFromCards([]Card{
	Defuse,
	Skip, Skip, Skip, Skip, Skip,
	Slap1x, Slap1x, Slap1x,
	Slap2x,
	SeeTheFuture, SeeTheFuture, SeeTheFuture,
	Shuffle, Shuffle,
	DrawFromTheBottom, DrawFromTheBottom,
	Cat, Cat, Cat,
})
