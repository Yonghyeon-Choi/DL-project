import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model

IMG_SIZE = 299

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


cnn_model = build_feature_extractor()

plot_model(cnn_model, to_file='cnn_model.png', show_shapes=True, dpi=320)
