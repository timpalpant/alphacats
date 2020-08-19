import argparse
import glob
import logging
import os
import shutil

import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping,
    TerminateOnNaN,
    ModelCheckpoint,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    concatenate,
    LSTM,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# These constants must be kept in sync with the Go code.
TF_GRAPH_TAG = "lstm"

MAX_HISTORY = 58
N_ACTION_FEATURES = 16
NUM_CARD_TYPES = 11
MAX_CARDS_IN_DRAW_PILE = 13
N_OUTPUTS = NUM_CARD_TYPES + MAX_CARDS_IN_DRAW_PILE + 1


def build_model(history_shape: tuple, hands_shape: tuple, drawpile_shape: tuple, policy_shape: int):
    logging.info("Building model")
    logging.info("History input shape: %s", history_shape)
    logging.info("Hands input shape: %s", hands_shape)
    logging.info("Draw pile input shape: %s", drawpile_shape)
    logging.info("Policy output shape: %s", policy_shape)

    # The history (LSTM) arm of the model.
    history_input = Input(name="history", shape=history_shape)
    lstm = Bidirectional(LSTM(64, return_sequences=False))(history_input)

    # The private hand arm of the model.
    hands_input = Input(name="hands", shape=hands_shape)

    # The draw pile arm of the model.
    drawpile_input = Input(name="drawpile", shape=drawpile_shape)

    # Concatenate and predict advantages.
    merged_inputs = concatenate([lstm, hands_input, drawpile_input])
    merged_hidden_1 = Dense(128, activation='relu')(merged_inputs)
    merged_hidden_2 = Dense(128, activation='relu')(merged_hidden_1)
    merged_hidden_3 = Dense(128, activation='relu')(merged_hidden_2)
    merged_hidden_4 = Dense(64, activation='relu')(merged_hidden_3)

    policy_hidden_1 = Dense(64, activation='relu')(merged_hidden_4)
    policy_output = Dense(policy_shape, activation='softmax', name='policy')(policy_hidden_1)

    value_hidden_1 = Dense(64, activation='relu')(merged_hidden_4)
    value_output = Dense(1, activation='linear', name='value')(value_hidden_1)

    model = Model(
        inputs=[history_input, hands_input, drawpile_input],
        outputs=[policy_output, value_output])
    model.compile(
        loss=['categorical_crossentropy', 'mean_squared_error'],
        optimizer=Adam(clipnorm=1.0),
        metrics=['mean_absolute_error'])
    return model


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


def load_data(filename: str):
    batch = np.load(filename)
    n_samples = len(batch["Y_value"])
    X_history = batch["X_history"].reshape((n_samples, MAX_HISTORY, N_ACTION_FEATURES))
    X_hands = batch["X_hands"].reshape((n_samples, 3*NUM_CARD_TYPES))
    X_drawpile = batch["X_drawpile"].reshape((n_samples, MAX_CARDS_IN_DRAW_PILE * NUM_CARD_TYPES))
    X = {"history": X_history, "hands": X_hands, "drawpile": X_drawpile}
    Y_policy = batch["Y_policy"].reshape((n_samples, N_OUTPUTS))
    Y_value = batch["Y_value"].reshape((n_samples, 1))
    Y = {"policy": Y_policy, "value": Y_value}
    return X, Y


def main():
    parser = argparse.ArgumentParser(description="Run training on a batch of advantages samples")
    parser.add_argument("input", help="Input with training data (npz)")
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

    X, y = load_data(args.input)

    history_shape = X["history"][0].shape
    hands_shape = X["hands"][0].shape
    drawpile_shape = X["drawpile"][0].shape
    policy_shape = y["policy"][0].shape[0]
    model = build_model(history_shape, hands_shape, drawpile_shape, policy_shape)
    print(model.summary())
    print("Input layer names:", [node.op.name for node in model.inputs])
    print("Output layer names:", [node.op.name for node in model.outputs])

    if args.initial_weights:
        logging.info("Loading initial weights from: %s", args.initial_weights)
        model.load_weights(args.initial_weights)

    history = model.fit(
        x=X,
        y=y,
        epochs=20,
        validation_split=0.1,
        use_multiprocessing=False,
        workers=4,
        max_queue_size=8,
        callbacks=[
            EarlyStopping(
                monitor='val_loss', min_delta=0.0001, patience=5,
                restore_best_weights=True),
            TerminateOnNaN(),
        ],
    )

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
