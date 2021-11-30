import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

MAX_SEQ_LENGTH = 10
NUM_FEATURES = 2048

frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

# Refer to the following tutorial to understand the significance of using `mask`:
# https://keras.io/api/layers/recurrent_layers/gru/
x = keras.layers.GRU(32, return_sequences=True)(
    frame_features_input, mask=mask_input
)
x = keras.layers.GRU(16)(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(8, activation="relu")(x)
output = keras.layers.Dense(6, activation="softmax")(x)
rnn_model = keras.Model([frame_features_input, mask_input], output)

plot_model(rnn_model, to_file='rnn_model.png', show_shapes=True, dpi=320)
