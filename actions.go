package alphacats

type Action int

const (
	DrawCard Action = iota

	ReplaceExplodingCat1
	ReplaceExplodingCat2
	ReplaceExplodingCat3
	ReplaceExplodingCat4
	ReplaceExplodingCat5
	ReplaceExplodingCatBottom
	ReplaceExplodingCatRandom

	PlayDefuseCard
	PlaySkipCard
	PlaySlap1xCard
	PlaySlap2xCard
	PlaySeeTheFutureCard
	PlayShuffleCard
	PlayDrawFromTheBottomCard
	PlayCatCard

	GiveDefuseCard
	GiveSkipCard
	GiveSlap1xCard
	GiveSlap2xCard
	GiveSeeTheFutureCard
	GiveShuffleCard
	GiveDrawFromTheBottomCard
	GiveCatCard
)

var playCardActions = map[Card]Action{
	Defuse:            PlayDefuseCard,
	Skip:              PlaySkipCard,
	Slap1x:            PlaySlap1xCard,
	Slap2x:            PlaySlap2xCard,
	SeeTheFuture:      PlaySeeTheFutureCard,
	Shuffle:           PlayShuffleCard,
	DrawFromTheBottom: PlayDrawFromTheBottomCard,
	Cat:               PlayCatCard,
}

func GetPlayActionForCard(card Card) Action {
	return playCardActions[card]
}

var giveCardActions = map[Card]Action{
	Defuse:            GiveDefuseCard,
	Skip:              GiveSkipCard,
	Slap1x:            GiveSlap1xCard,
	Slap2x:            GiveSlap2xCard,
	SeeTheFuture:      GiveSeeTheFutureCard,
	Shuffle:           GiveShuffleCard,
	DrawFromTheBottom: GiveDrawFromTheBottomCard,
	Cat:               GiveCatCard,
}

func GetGiveActionForCard(card Card) Action {
	return giveCardActions[card]
}
