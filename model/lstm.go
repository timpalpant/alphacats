// Package model implements an LSTM-based network model for use in DeepCFR or MCTS.
// As input, the model takes the public game history (one-hot encoded),
// as well as the player's current hand (one-hot encoded) and predicts the
// advantages for each possible action in this infoset.
package model

import (
	"bytes"
	"encoding/gob"
	"expvar"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/golang/glog"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/deepcfr"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
	"github.com/timpalpant/alphacats/gamestate"
)

const (
	graphTag    = "lstm"
	outputLayer = "output/BiasAdd"
)

var (
	samplesPredicted = expvar.NewInt("num_predicted_samples")
	batchesPredicted = expvar.NewInt("num_predicted_batches")
)

// tfConfig is tf.ConfigProto(
//   gpu_options=tf.GPUOptions(allow_growth=True)
// ).SerializeToString()
var tfConfig = []byte{50, 2, 32, 1}

type Params struct {
	BatchSize              int
	OutputDir              string
	NumEncodingWorkers     int
	MaxTrainingDataWorkers int
	MaxInferenceBatchSize  int
	NumPredictionWorkers   int
}

// LSTM is a model for AlphaCats to be used with DeepCFR
// and implements deepcfr.Model.
type LSTM struct {
	params Params
	iter   int
}

func NewLSTM(p Params) *LSTM {
	return &LSTM{
		params: p,
		iter:   1,
	}
}

// Train implements deepcfr.Model.
func (m *LSTM) Train(buffer deepcfr.Buffer) deepcfr.TrainedModel {
	samples := buffer.GetSamples()
	tuples := make([]*deepcfr.RegretSample, len(samples))
	for i, s := range samples {
		x := *s.(*deepcfr.RegretSample)
		tuples[i] = &x
	}

	glog.Infof("Training network with %d samples", len(samples))
	// Save training data to disk in a tempdir.
	tmpDir, err := ioutil.TempDir(m.params.OutputDir, "training-data-")
	if err != nil {
		glog.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	glog.Infof("Saving training data to: %v", tmpDir)
	if err := saveTrainingData(tuples, tmpDir, m.params.BatchSize, m.params.MaxTrainingDataWorkers); err != nil {
		glog.Fatal(err)
	}

	// Shell out to Python to train the network.
	outputDir := filepath.Join(m.params.OutputDir, fmt.Sprintf("model_%08d", m.iter))
	args := []string{"model/train.py", tmpDir, outputDir}
	cmd := exec.Command("python", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	glog.Infof("Running: %v", cmd.Args)
	start := time.Now()
	if err := cmd.Run(); err != nil {
		glog.Fatal(err)
	}

	elapsed := time.Since(start)
	sps := float64(len(samples)) / elapsed.Seconds()
	glog.V(1).Infof("Finished training (took %v, %v samples/sec)",
		elapsed, sps)
	m.iter++

	// Load trained model.
	trained, err := LoadTrainedLSTM(outputDir, m.params)
	if err != nil {
		glog.Fatal(err)
	}

	return trained
}

func (m *LSTM) UnmarshalBinary(buf []byte) error {
	r := bytes.NewReader(buf)
	dec := gob.NewDecoder(r)

	if err := dec.Decode(&m.params); err != nil {
		return err
	}

	if err := dec.Decode(&m.iter); err != nil {
		return err
	}

	return nil
}

func (m *LSTM) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(m.params); err != nil {
		return nil, err
	}

	if err := enc.Encode(m.iter); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

type TrainedLSTM struct {
	dir    string
	params Params
	model  *tf.SavedModel
	reqsCh chan *predictionRequest
}

func LoadTrainedLSTM(dir string, params Params) (*TrainedLSTM, error) {
	opts := &tf.SessionOptions{Config: tfConfig}
	model, err := tf.LoadSavedModel(dir, []string{graphTag}, opts)
	if err != nil {
		return nil, err
	}

	m := &TrainedLSTM{
		dir:    dir,
		params: params,
		model:  model,
		reqsCh: make(chan *predictionRequest, 1),
	}
	go m.bgPredictionHandler()
	return m, nil
}

// When serializing, we just serialize the path to the model already on disk.
func (m *TrainedLSTM) GobDecode(buf []byte) error {
	r := bytes.NewReader(buf)
	dec := gob.NewDecoder(r)

	var dir string
	if err := dec.Decode(&dir); err != nil {
		return err
	}

	var params Params
	if err := dec.Decode(&params); err != nil {
		return err
	}

	model, err := LoadTrainedLSTM(dir, params)
	if err != nil {
		return err
	}

	*m = *model
	return nil
}

func (m *TrainedLSTM) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(m.dir); err != nil {
		return nil, err
	}

	if err := enc.Encode(m.params); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func (m *TrainedLSTM) Close() {
	close(m.reqsCh)
}

const (
	tfHistorySize  = 4 * gamestate.MaxNumActions * numActionFeatures
	tfHandSize     = 4 * cards.NumTypes
	tfDrawPileSize = 4 * maxCardsInDrawPile * cards.NumTypes
)

var predictionRequestPool = sync.Pool{
	New: func() interface{} {
		return &predictionRequest{
			resultCh: make(chan []float32, 1),
		}
	},
}

// Predict implements deepcfr.TrainedModel.
func (m *TrainedLSTM) Predict(infoSet cfr.InfoSet, nActions int) []float32 {
	is := infoSet.(*alphacats.AbstractedInfoSet)
	if len(is.AvailableActions) != nActions {
		panic(fmt.Errorf("InfoSet has %d actions but expected %d: %v",
			len(is.AvailableActions), nActions, is.AvailableActions))
	}

	tfHistory := make([]byte, tfHistorySize)
	encodeHistoryTF(is.PublicHistory, tfHistory)
	tfHands := make([]byte, 3*tfHandSize)
	encodeHandTF(is.Hand, tfHands)
	encodeHandTF(is.P0PlayedCards, tfHands[tfHandSize:])
	encodeHandTF(is.P1PlayedCards, tfHands[2*tfHandSize:])
	tfDrawPile := make([]byte, tfDrawPileSize)
	encodeDrawPileTF(is.DrawPile)
	req := predictionRequestPool.Get().(*predictionRequest)
	req.history = tfHistory
	req.hands = tfHands

	m.reqsCh <- req
	advantages := <-req.resultCh
	predictionRequestPool.Put(req)

	result := make([]float32, nActions)
	for i, action := range is.AvailableActions {
		switch action.Type {
		case gamestate.DrawCard:
			// First position is always the advantages of ending turn by drawing a card,
			// since this corresponds to the "Unknown" card enum.
			result[i] = advantages[0]
		case gamestate.PlayCard, gamestate.GiveCard:
			// Next 9 positions correspond to playing/giving each card type.
			result[i] = advantages[action.Card]
		case gamestate.InsertExplodingKitten:
			// Remaining correspond to inserting cat at each position.
			result[i] = advantages[cards.NumTypes+int(action.PositionInDrawPile)]
		default:
			panic(fmt.Errorf("unsupported action: %v", action))
		}
	}

	return advantages
}

type predictionRequest struct {
	history  []byte
	hands    []byte
	resultCh chan []float32
}

// Handles prediction requests, attempting to batch all pending requests.
func (m *TrainedLSTM) bgPredictionHandler() {
	// No buffers here because we are collecting batches and we want the batch to be as
	// large as possible until we are ready to process it.
	outputCh := make(chan *batchPredictionRequest)
	defer close(outputCh)
	glog.V(1).Infof("Starting %d batch prediction workers",
		m.params.NumPredictionWorkers)
	for i := 0; i < m.params.NumPredictionWorkers; i++ {
		go handleBatchPredictions(m.model, outputCh)
	}

	encodeCh := make(chan []*predictionRequest)
	defer close(encodeCh)
	// Multiple encoder threads because NewTensor is slow
	// and we don't want to be bottlenecked on it. This will mean too-small batches
	// for the first few, but we expect that at steady-state we will be rate-limited
	// by predictions on the GPU, so the batches will still be large (just buffered).
	glog.V(1).Infof("Starting %d batch encoding workers",
		m.params.NumEncodingWorkers)
	for i := 0; i < m.params.NumEncodingWorkers; i++ {
		go handleEncoding(m.model, encodeCh, outputCh)
	}

	for {
		// Wait for first request so we know we have at least one.
		req, ok := <-m.reqsCh
		if !ok {
			return
		}

		// Drain any additional requests as long as we're waiting for an encoding spot.
		batch := []*predictionRequest{req}
	Loop:
		for {
			select {
			case req, ok := <-m.reqsCh:
				if !ok {
					encodeCh <- batch
					return
				}

				batch = append(batch, req)
				if len(batch) >= m.params.MaxInferenceBatchSize {
					encodeCh <- batch
					break Loop
				}
			case encodeCh <- batch:
				break Loop
			}
		}
	}
}

func concat(batch []*predictionRequest) (histories, hands, drawPiles []byte) {
	historyLen := 0
	handsLen := 0
	drawPilesLen := 0
	for _, req := range batch {
		historyLen += len(req.history)
		handsLen += len(req.hands)
		drawPilesLen += len(req.drawPiles)
	}

	histories = make([]byte, 0, historyLen)
	hands = make([]byte, 0, handsLen)
	drawPiles = make([]byte, 0, drawPilesLen)
	for _, req := range batch {
		histories = append(histories, req.history...)
		hands = append(hands, req.hands...)
		drawPiles = append(drawPiles, req.drawPiles...)
	}

	return histories, hands, drawPiles
}

func handleEncoding(model *tf.SavedModel, batchCh chan []*predictionRequest, outputCh chan *batchPredictionRequest) {
	for batch := range batchCh {
		// TODO: Shapes should be passed in to avoid coupling here.
		historiesBuf, handsBuf, drawPilesBuf := concat(batch)
		historiesReader := bytes.NewReader(historiesBuf)
		historiesShape := []int64{int64(len(batch)), gamestate.MaxNumActions, numActionFeatures}
		historyTensor, err := tf.ReadTensor(tf.Float, historiesShape, historiesReader)
		if err != nil {
			glog.Fatal(err)
		}

		handsReader := bytes.NewReader(handsBuf)
		handsShape := []int64{int64(len(batch)), int64(cards.NumTypes)}
		handTensor, err := tf.ReadTensor(tf.Float, handsShape, handsReader)
		if err != nil {
			glog.Fatal(err)
		}

		drawPilesReader := bytes.NewReader(drawPilesBuf)
		drawPilesShape := []int64{int64(len(batch)), int64(maxCardsInDrawPile * cards.NumTypes)}
		drawPilesTensor, err := tf.ReadTensor(tf.Float, drawPilesShape, drawPileReader)
		if err != nil {
			glog.Fatal(err)
		}

		outputCh <- &batchPredictionRequest{
			history:   historyTensor,
			hands:     handTensor,
			drawPiles: drawPilesTensor,
			batch:     batch,
		}
	}
}

type batchPredictionRequest struct {
	history   *tf.Tensor
	hands     *tf.Tensor
	drawPiles *tf.Tensor
	batch     []*predictionRequest
}

func handleBatchPredictions(model *tf.SavedModel, reqCh chan *batchPredictionRequest) {
	defer model.Session.Close()
	for req := range reqCh {
		resultTensor := predictBatch(model, req.history, req.hands, req.drawPiles)
		result := resultTensor.Value().([][]float32)
		for i, req := range req.batch {
			req.resultCh <- result[i]
		}
		samplesPredicted.Add(int64(len(req.batch)))
		batchesPredicted.Add(1)
	}
}

func predictBatch(model *tf.SavedModel, history, hands, drawPiles *tf.Tensor) *tf.Tensor {
	result, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("history").Output(0):  history,
			model.Graph.Operation("hands").Output(0):    hands,
			model.Graph.Operation("drawpile").Output(0): drawPiles,
		},
		[]tf.Output{
			model.Graph.Operation(outputLayer).Output(0),
		},
		nil,
	)

	if err != nil {
		glog.Fatal(err)
	}

	return result[0]
}

func init() {
	gob.Register(&LSTM{})
	gob.Register(&TrainedLSTM{})
}
