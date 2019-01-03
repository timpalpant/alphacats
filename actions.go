package alphacats

type Action int

const (
	DrawCard Action = iota

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

func ApplyAction(infoSet InfoSet, action Action) InfoSet {
	switch action {
	case DrawCard:
		// We incorporate top card from draw pile into our hand.
	case PlayDefuseCard:
		infoSet.OurHand[Defuse]--
		// Choose where to place exploding cat in the draw pile.
	case PlaySkipCard:
		infoSet.OurHand[Skip]--
	case PlaySlap1xCard:
		infoSet.OurHand[Slap1x]--
	case PlaySlap2xCard:
		infoSet.OurHand[Slap2x]--
	case PlaySeeTheFutureCard:
		infoSet.OurHand[SeeTheFuture]--
	case PlayShuffleCard:
		infoSet.OurHand[Shuffle]--
	case PlayDrawFromTheBottomCard:
		infoSet.OurHand[DrawFromTheBottom]--
		// We incorporate bottom card from draw pile into our hand.
	case PlayCatCard:
		infoSet.OurHand[Cat]--
		// Other player must give choose a card to give us.
	case GiveDefuseCard:
		infoSet.OurHand[Defuse]--
		infoSet.OpponentHand[Defuse]++
	case GiveSkipCard:
		infoSet.OurHand[Skip]--
		infoSet.OpponentHand[Skip]++
	case GiveSlap1xCard:
		infoSet.OurHand[Slap1x]--
		infoSet.OpponentHand[Slap1x]++
	case GiveSlap2xCard:
		infoSet.OurHand[Slap2x]--
		infoSet.OpponentHand[Slap2x]++
	case GiveSeeTheFutureCard:
		infoSet.OurHand[SeeTheFuture]--
		infoSet.OpponentHand[SeeTheFuture]++
	case GiveShuffleCard:
		infoSet.OurHand[Shuffle]--
		infoSet.OpponentHand[Shuffle]++
	case GiveDrawFromTheBottomCard:
		infoSet.OurHand[DrawFromTheBottom]--
		infoSet.OpponentHand[DrawFromTheBottom]++
	case GiveCatCard:
		infoSet.OurHand[Cat]--
		infoSet.OpponentHand[Cat]++
	default:
		panic("invalid action")
	}

	return infoSet
}
