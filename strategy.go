package alphacats

type Strategy struct {
	// For each info set => available actions => probability we take it.
	actions map[InfoSet]map[Action]float64
}
