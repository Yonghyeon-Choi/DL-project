from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
import cv2
import os

"""
## Define hyperparameters
"""

LEARNING_RATE = 3e-4
BATCH_SIZE = 80
EPOCHS = 20

IMG_SIZE = 299
MAX_SEQ_LENGTH = 1
NUM_FEATURES = 2048

"""
## Data preparation
"""

train_df = pd.read_csv(os.path.join("csv", "train.csv"))
val_df = pd.read_csv(os.path.join("csv", "val.csv"))
test_df = pd.read_csv(os.path.join("csv", "test.csv"))

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for validation: {len(val_df)}")
print(f"Total videos for testing: {len(test_df)}")


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


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


feature_extractor = build_feature_extractor()
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(tqdm(video_paths)):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    return (frame_features, frame_masks), labels


print()
print("prepare train data")
train_data, train_labels = prepare_all_videos(train_df, os.path.join("video", "train"))
print()
print("prepare val data")
val_data, val_labels = prepare_all_videos(val_df, os.path.join("video", "val"))
print()
print("prepare test data")
test_data, test_labels = prepare_all_videos(test_df, os.path.join("video", "test"))

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")


def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

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
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=Adam(learning_rate=LEARNING_RATE)
    )
    return rnn_model


this_time = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d-%H-%M-%S")
result = open(os.path.join("result", this_time+".txt"),
              mode="wt", encoding="utf-8")


def run_experiment():
    filepath = os.path.join("save", "video_classifier", this_time)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_data=([val_data[0], val_data[1]], val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[checkpoint]
    )
    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    result.write(f"Test accuracy: {round(accuracy * 100, 2)}%\n")
    return history, seq_model


history, sequence_model = run_experiment()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], 'b')
plt.plot(history.history['val_accuracy'], 'r')
plt.xlabel('Epoch '+str(EPOCHS))
plt.ylabel('Accuracy')
plt.legend(['Accuracy', 'Val_Accuracy'], loc='best')
plt.savefig(os.path.join('result', 'Accuracy'+this_time+'.png'), dpi=200)
plt.clf()

plt.title('Loss')
plt.plot(history.history['loss'], '--b')
plt.plot(history.history['val_loss'], '--r')
plt.xlabel('Epoch '+str(EPOCHS))
plt.ylabel('Loss')
plt.legend(['Loss', 'Val_Loss'], loc='best')
plt.savefig(os.path.join('result', 'Loss'+this_time+'.png'), dpi=200)


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("video", "test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        result.write(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%\n")
    return frames


test_videos = test_df["video_name"].values.tolist()
test_tags = test_df["tag"].values.tolist()
for i in range(len(test_videos)):
    print()
    result.write("\n")
    print(f"{i+1} Test video path: {test_videos[i]}")
    result.write(f"{i} Test video path: {test_videos[i]}\n")
    print(f"  Actual: {test_tags[i]}")
    result.write(f"  Actual: {test_tags[i]}\n")
    test_frames = sequence_prediction(test_videos[i])

result.close()
