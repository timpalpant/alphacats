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
    BatchNormalization,
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
import matplotlib.pyplot as plt
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
    merged_hidden_1 = Dense(128, activation='relu')(merged)
    merged_hidden_2 = Dense(128, activation='relu')(merged_hidden_1)
    merged_hidden_3 = Dense(128, activation='relu')(merged_hidden_2)
    normalization = BatchNormalization()(merged_hidden_3)
    advantages_output = Dense(1, activation='linear', name='output')(normalization)

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
        workers=4,
        max_queue_size=8,
        callbacks=[
            EarlyStopping(
                monitor='val_loss', min_delta=0.001, patience=3,
                restore_best_weights=True),
            TerminateOnNaN(),
        ],
    )

    return model, history


def plot_metrics(history, output):
    plt.figure()
    for metric in ['loss', 'val_loss']:
        epochs = np.arange(len(history.history[metric])) + 1
        plt.plot(epochs, history.history[metric], label=metric)
    plt.xticks(epochs)
    plt.legend()
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(output)


def main():
    parser = argparse.ArgumentParser(description="Run training on a batch of advantages samples")
    parser.add_argument("input", help="Input directory with batches of training data (npz)")
    parser.add_argument("output", help="Directory to save trained model to")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="Fraction of data to hold out for validation / early-stopping")
    parser.add_argument("--initial_weights", help="Load initial weights from saved model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = tf.ConfigProto(
      gpu_options=tf.GPUOptions(allow_growth=True),
    )
    sess = tf.Session(config=config)
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

    if args.initial_weights:
        model.load_weights(args.initial_weights)

    model, history = train(model, data, val_data)

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    logging.info("Saving model to %s", args.output)
    builder = tf.saved_model.builder.SavedModelBuilder(args.output)
    # Tag the model, required for Go
    builder.add_meta_graph_and_variables(sess, [TF_GRAPH_TAG])
    builder.save()
    # Save keras model weights for re-initialization on next iteration.
    model.save_weights(os.path.join(args.output, "weights.h5"))
    plot_metrics(history, os.path.join(args.output, "metrics.pdf"))


if __name__ == "__main__":
  main()
