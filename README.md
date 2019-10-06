AlphaCats
=========

AlphaCats was a failed attempt to solve the game of [Exploding Kittens](https://explodingkittens.com) using [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164). AlphaCats is built around the [`go-cfr`](https://github.com/timpalpant/go-cfr) package.

Due to the depth of the game tree, external sampling is intractable, and other forms of MC-CFR sampling (such as outcome sampling), led to high-variance samples and a model that struggled to converge.

Future areas of investigation could include variance-reduction and improved sampling techniques.

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)
[![GoDoc](https://godoc.org/github.com/timpalpant/go-cfr?status.svg)](http://godoc.org/github.com/timpalpant/go-cfr)

# Usage

`cmd/alphacats` is the main driver binary. CFR iteration can be launched with:

```
./cmd/alphacats/alphacats -logtostderr \
    -decktype core -cfrtype deep -iter 10 \
    -sampling.num_sampling_threads 5000 \
    -sampling.max_num_actions 2 \
    -sampling.exploration_eps 1.0 \
    -deepcfr.traversals_per_iter 10000 \
    -deepcfr.buffer.size 10000000 \
    -deepcfr.model.num_encoding_workers 4 \
    -deepcfr.model.batch_size 10000 \
    -deepcfr.model.max_inference_batch_size 10000 \
    -output_dir output -v 1 2>&1 | tee run.log
```

This will run DeepCFR with a reservoir buffer of size 10 million, and sample the
game tree using [robust sampling](https://arxiv.org/abs/1812.10607) with K=2.

Tabular CFR can also be launched with `-cfrtype tabular`. It requires a large amount
of memory and therefore a smaller test game can be selected with `-decktype test`.
Tabular CFR is not thread-safe and must be run with `-sampling.num_sampling_threads 1`.

# Model

The underlying model used in AlphaCats is an LSTM over the game history that
feeds forward into a deep fully connected network.

```
# The history (LSTM) arm of the model.
history_input = Input(name="history", shape=history_shape)
lstm = Bidirectional(CuDNNLSTM(32, return_sequences=False))(history_input)

# The private hand arm of the model.
hands_input = Input(name="hands", shape=hands_shape)

# Concatenate and predict advantages.
merged_inputs = concatenate([lstm, hands_input])
merged_hidden_1 = Dense(128, activation='relu')(merged_inputs)
merged_hidden_2 = Dense(128, activation='relu')(merged_hidden_1)
merged_hidden_3 = Dense(128, activation='relu')(merged_hidden_2)
merged_hidden_4 = Dense(64, activation='relu')(merged_hidden_3)
merged_hidden_5 = Dense(64, activation='relu')(merged_hidden_4)
normalization = BatchNormalization()(merged_hidden_5)
advantages_output = Dense(N_OUTPUTS, activation='linear', name='output')(normalization)

model = Model(
    inputs=[history_input, hands_input],
    outputs=[advantages_output])
model.compile(
    loss='mean_squared_error',
    optimizer=Adam(clipnorm=1.0),
    metrics=['mean_absolute_error'])
```

See `model/train.py` for the training script. During training, samples are first
generated using a go-cfr sampler, saved to `*.npz` files, and then loaded by the script in minibatches.
The resulting model is saved in TensorFlow format, and loaded for inference (see `model/lstm.go`).
