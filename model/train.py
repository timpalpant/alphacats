import argparse
import glob
import logging
import os
import shutil

from keras import backend as K
from keras.callbacks import (
    EarlyStopping,
    TerminateOnNaN,
    ModelCheckpoint,
)
from keras.layers import (
    concatenate,
    CuDNNLSTM,
    Dense,
    Dropout,
    Input,
    multiply,
)
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.utils import Sequence
import numpy as np
import tensorflow as tf


# These constants must be kept in sync with the Go code.
TF_GRAPH_TAG = "lstm"

MAX_HISTORY = 48
N_ACTION_FEATURES = 59
NUM_CARD_TYPES = 10


class TrainingSequence(Sequence):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        logging.debug("Loading batch %d", idx)
        batch = np.load(self.batches[idx])
        n_samples = len(batch["sample_weight"])
        X_history = batch["X_history"].reshape((n_samples, MAX_HISTORY, N_ACTION_FEATURES))
        X_hand = batch["X_hand"].reshape((n_samples, NUM_CARD_TYPES))
        X_action = batch["X_action"].reshape((n_samples, N_ACTION_FEATURES))
        X = {"history": X_history, "hand": X_hand, "action": X_action}
        y = batch["y"].reshape((n_samples, 1))
        return X, y, batch["sample_weight"]


def build_model(history_shape: tuple, hand_shape: tuple, action_shape: tuple, output_shape: int):
    logging.info("Building model")
    logging.info("History input shape: %s", history_shape)
    logging.info("Hand input shape: %s", hand_shape)
    logging.info("Action input shape: %s", action_shape)
    logging.info("Output shape: %s", output_shape)

    # The history (LSTM) arm of the model.
    history_input = Input(name="history", shape=history_shape)
    lstm = Bidirectional(CuDNNLSTM(128, return_sequences=False))(history_input)

    # The private hand arm of the model.
    hand_input = Input(name="hand", shape=hand_shape)

    # The action we are evaluating.
    action_input = Input(name="action", shape=action_shape)

    # Concatenate and predict advantages.
    merged = concatenate([lstm, hand_input, action_input])
    merged_dropout_1 = Dropout(0.3)(merged)
    merged_hidden_1 = Dense(128, activation='relu')(merged_dropout_1)
    merged_dropout_2 = Dropout(0.3)(merged_hidden_1)
    merged_hidden_2 = Dense(128, activation='relu')(merged_dropout_2)
    merged_dropout_3 = Dropout(0.3)(merged_hidden_2)
    merged_hidden_3 = Dense(128, activation='relu')(merged_dropout_3)
    advantages_output = Dense(1, activation='linear', name='output')(merged_hidden_3)

    model = Model(
        inputs=[history_input, hand_input, action_input],
        outputs=[advantages_output])
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_absolute_error'])
    return model


def train(model, data, val_data):
    history = model.fit_generator(
        data,
        epochs=50,
        validation_data=val_data,
        use_multiprocessing=False,
        workers=2,
        max_queue_size=2,
        callbacks=[
            EarlyStopping(
                monitor='val_loss', min_delta=0.001, patience=3,
                restore_best_weights=True),
            TerminateOnNaN(),
        ],
    )

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Run training on a batch of advantages samples")
    parser.add_argument("input", help="Input directory with batches of training data (npz)")
    parser.add_argument("output", help="Directory to save trained model to")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="Fraction of data to hold out for validation / early-stopping")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    sess = tf.Session()
    K.set_session(sess)

    batches = sorted(glob.glob(os.path.join(args.input, "batch_*.npz")))
    logging.info("Found %d batches in %s", len(batches), args.input)
    val_n = int(args.validation_split * len(batches))
    logging.info("Using %d batches for validation", val_n)
    data = TrainingSequence(batches[val_n:])
    val_data = TrainingSequence(batches[:val_n])

    X, y, _ = data[0]
    history_shape = X["history"][0].shape
    hand_shape = X["hand"][0].shape
    action_shape = X["action"][0].shape
    output_shape = y[0].shape[0]
    model = build_model(history_shape, hand_shape, action_shape, output_shape)
    print(model.summary())
    logging.info("All nodes in TF graph:")
    for node in tf.get_default_graph().as_graph_def().node:
        logging.info(node.name)

    model, history = train(model, data, val_data)

    if os.path.exists(args.output):
      shutil.rmtree(args.output)

    logging.info("Saving model to %s", args.output)
    builder = tf.saved_model.builder.SavedModelBuilder(args.output)
    # Tag the model, required for Go
    builder.add_meta_graph_and_variables(sess, [TF_GRAPH_TAG])
    builder.save()


if __name__ == "__main__":
  main()
