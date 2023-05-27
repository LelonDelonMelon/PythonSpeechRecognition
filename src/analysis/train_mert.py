import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import librosa
import json
from sklearn.model_selection import train_test_split

# Set the path to the Librispeech dataset
trainData_path = "LibriSpeech/train-clean"
testData_path = "LibriSpeech/test-clean"
json_path_train = "src/json/train_data.json"
json_path_test = "src/json/test_data.json"

# Set the desired user IDs
user_ids = [19, 26, 27, 32, 39, 40, 78, 83, 196, 405]
num_classes = len(user_ids)
MAX_LENGTH = 1704  # Maximum length for padding. This is the max value in train set.

# Set the parameters for audio feature extraction
sample_rate = 16000  # Adjust according to your dataset
n_mfcc = 13  # Number of MFCC coefficients
frame_length = 0.025  # Length of each frame in seconds
frame_stride = 0.01  # Length of stride between frames in seconds

# Preprocess audio data and extract features (MFCCs)
def preprocess_audio(audio_path, sample_rate=16000, n_mfcc=13, frame_length=0.025, frame_stride=0.01):
    # Load audio file
    audio, _ = librosa.load(audio_path, sr=sample_rate)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc,
                                 hop_length=int(frame_stride * sample_rate),
                                 n_fft=int(frame_length * sample_rate))

    # Normalize MFCCs (optional)
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    # Transpose MFCCs to have the shape (num_frames, num_mfcc, 1)
    mfccs = mfccs.T

    return mfccs

# Prepare the training and validation data
def prepare_data(pathToDataset, jsonPath):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    num_mfcc = 13

    for user_id in user_ids:
        user_dir = os.path.join(pathToDataset, f"{user_id}")
        data["mapping"].append(user_id)
        for file_name in os.listdir(user_dir):
            audio_path = os.path.join(user_dir, file_name)
            features = preprocess_audio(audio_path)
            if features is not None:  # Skip audio files with errors
                if len(features) < MAX_LENGTH:
                    features = np.pad(features, ((0, MAX_LENGTH - len(features)), (0, 0)), mode='constant')
                else:
                    features = features[:MAX_LENGTH]
                data["mfcc"].append(features.tolist())
                data["labels"].append(user_ids.index(user_id))

    with open(jsonPath, "w") as fp:
        json.dump(data, fp, indent=4)

# Build the model
def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(input_shape[1:])),
        layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(256, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(64, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(num_classes, activation="softmax")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


# Train the model
def train_model(model, training_data, training_labels, validation_data, validation_labels, batch_size=32, epochs=50):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    model.fit(
        training_data,
        training_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_data, validation_labels),
        callbacks=[early_stopping]
    )

# Evaluate the model
def evaluate_model_accuracy(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, test_data, test_labels, class_labels):
    # Make predictions on the test data
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    
    predicted_user_ids = [user_ids[label] for label in predicted_labels]
    actual_user_ids = [user_ids[label] for label in test_labels]
    
    # Display predicted user IDs and actual user IDs
    print("Predicted User IDs:", predicted_user_ids)
    print("Actual User IDs:", actual_user_ids)


    # to see the confusion matrix of test sets evaluation.
    # plot_confusiton_matrix(actual_user_ids,predicted_user_ids)


# Plots the confusion matrix of model.
def plot_confusiton_matrix(actual_user_ids,predicted_user_ids):
    # Create a confusion matrix
    cm = confusion_matrix(actual_user_ids, predicted_user_ids)

    # Display confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=user_ids, yticklabels=user_ids)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Display classification report
    print(classification_report(actual_user_ids,predicted_user_ids))

# Load the data from JSON file
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    # Convert lists into numpy arrays
    inputs = np.array(data["mfcc"], dtype=np.float32)
    targets = np.array(data["labels"])
    return inputs, targets

# Split the data into training and validation sets
def split_data():
    inputs, targets = load_data(json_path_train)
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=0.2)
    return inputs_train, inputs_val, targets_train, targets_val


# Prepare json files for train and test sets
def prepare_jsons():
    prepare_data(trainData_path, json_path_train)
    prepare_data(testData_path,json_path_test)


# Train and save the model
def train_and_save_model():
    
    inputs_train, inputs_validation, targets_train, targets_validation = split_data()
    model = build_model(inputs_train.shape, num_classes)
    train_model(model, inputs_train, targets_train, inputs_validation, targets_validation)
    model.summary()
    model.save("src/model/cnn.h5")

# Test the trained model
# Achieved 90% Accuracy
def test_model():
    
    inputs_test, targets_test = load_data(json_path_test)
    model = tf.keras.models.load_model("src/model/cnn.h5")
    print("Shape of inputs_test:", inputs_test.shape)
    print("Shape of targets_test:", targets_test.shape)
    evaluate_model_accuracy(model, inputs_test, targets_test)
    class_labels = [str(user_id) for user_id in user_ids]  # Convert user_ids to string labels
    evaluate_model(model, inputs_test, targets_test, class_labels)


# Main function
def main():
    test_model()

main()
