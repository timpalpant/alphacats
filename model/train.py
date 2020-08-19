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
    LSTM,
    Dense,
    Input,
)
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import (
    plot_model,
    Sequence,
)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# These constants must be kept in sync with the Go code.
TF_GRAPH_TAG = "lstm"

MAX_HISTORY = 58
N_ACTION_FEATURES = 16
NUM_CARD_TYPES = 11
MAX_CARDS_IN_DRAW_PILE = 13
N_OUTPUTS = NUM_CARD_TYPES + MAX_CARDS_IN_DRAW_PILE + 1


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
        X_hands = batch["X_hands"].reshape((n_samples, 3*NUM_CARD_TYPES))
        X_drawpile = batch["X_drawpile"].reshape((n_samples, MAX_CARDS_IN_DRAW_PILE * NUM_CARD_TYPES))
        X = {"history": X_history, "hands": X_hands, "drawpile": X_drawpile}
        y = batch["y"].reshape((n_samples, N_OUTPUTS))
        return X, y, batch["sample_weight"]


def build_model(history_shape: tuple, hands_shape: tuple, drawpile_shape: tuple, output_shape: int):
    logging.info("Building model")
    logging.info("History input shape: %s", history_shape)
    logging.info("Hands input shape: %s", hands_shape)
    logging.info("Draw pile input shape: %s", drawpile_shape)
    logging.info("Output shape: %s", output_shape)

    # The history (LSTM) arm of the model.
    history_input = Input(name="history", shape=history_shape)
    lstm = Bidirectional(LSTM(64, return_sequences=False))(history_input)

    # The private hand arm of the model.
    hands_input = Input(name="hands", shape=hands_shape)
    hands_hidden_1 = Dense(64, activation='relu')(hands_input)

    # The draw pile arm of the model.
    drawpile_input = Input(name="drawpile", shape=drawpile_shape)
    drawpile_hidden_1 = Dense(64, activation='relu')(drawpile_input)

    # Concatenate and predict advantages.
    merged_inputs = concatenate([lstm, hands_hidden_1, drawpile_hidden_1])
    merged_hidden_1 = Dense(128, activation='relu')(merged_inputs)
    merged_hidden_2 = Dense(128, activation='relu')(merged_hidden_1)
    merged_hidden_3 = Dense(128, activation='relu')(merged_hidden_2)
    merged_hidden_4 = Dense(64, activation='relu')(merged_hidden_3)
    merged_hidden_5 = Dense(64, activation='relu')(merged_hidden_4)
    normalization = BatchNormalization()(merged_hidden_5)
    advantages_output = Dense(N_OUTPUTS, activation='linear', name='output')(normalization)

    model = Model(
        inputs=[history_input, hands_input, drawpile_input],
        outputs=[advantages_output])
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(clipnorm=1.0),
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
    hands_shape = X["hands"][0].shape
    drawpile_shape = X["drawpile"][0].shape
    output_shape = y[0].shape[0]
    model = build_model(history_shape, hands_shape, drawpile_shape, output_shape)
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
    plot_model(model, to_file=os.path.join(args.output, 'model.pdf'),
               show_layer_names=False, show_shapes=True)
    plot_metrics(history, os.path.join(args.output, "metrics.pdf"))


if __name__ == "__main__":
  main()
