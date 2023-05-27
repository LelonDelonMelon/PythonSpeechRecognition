import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
# Set the path to the Librispeech dataset

trainData_path = "LibriSpeech/train-clean"
testData_path = "LibriSpeech/test-clean"
json_path_train = "src/json/train_data.json"
json_path_test = "src/json/test_data.json"
# Set the desired user IDs
user_ids = [19, 26, 32, 27, 39, 78, 40, 405, 83, 196]
num_classes = len(user_ids)

# Set the parameters for audio feature extraction
sample_rate = 16000  # Adjust according to your dataset
n_mfcc = 13  # Number of MFCC coefficients
frame_length = 0.025  # Length of each frame in seconds
frame_stride = 0.01  # Length of stride between frames in seconds
import librosa
import json
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

def prepare_data(pathToDataset,jsonPath):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    num_mfcc = 13

    for user_id in user_ids:
        user_dir = os.path.join(pathToDataset, f"{user_id}")
        for file_name in os.listdir(user_dir):
            audio_path = os.path.join(user_dir, file_name)
            features = preprocess_audio(audio_path)
            if features is not None:  # Skip audio files with errors
                data["mfcc"].append(features.tolist())
                data["labels"].append(user_ids.index(user_id))
    
    max_length = max(len(features) for features in data["mfcc"])
    mfcc_array = np.zeros((len(data["mfcc"]), max_length, num_mfcc))  # Initialize array

    for i, features in enumerate(data["mfcc"]):
        mfcc_array[i, :len(features)] = features  # Assign features to array

    data["mfcc"] = mfcc_array.tolist()  # Convert back to a list of lists
    
    with open(jsonPath ,"w") as fp:
            json.dump(data,fp, indent=4)


# Build the model
import tensorflow as tf

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([    
        
        
        
        
        layers.Flatten(input_shape=(input_shape[1],input_shape[2])),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
        
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model


# Train the model
def train_model(model, training_data, training_labels, validation_data, validation_labels, batch_size=32, epochs=10):
    model.fit(
        training_data,
        training_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_data, validation_labels)
    )
    # Evaluate the model
def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data = json.load(fp)
    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"], dtype= np.float32)
    targets = np.array(data["labels"])
    return inputs, targets


def prepareJsons():
    prepare_data(trainData_path,json_path_train)
    prepare_data(testData_path,json_path_test)

def split_data():

    from sklearn.model_selection import train_test_split

    # Load data
    inputs, targets = load_data(json_path_train)
    # Train Test Split
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs,
                                                                            targets,
                                                                            test_size= 0.2)
    return inputs_train, inputs_val, targets_train, targets_val


def train_and_save_model():    
    # Prepare Train JSONs
    prepareJsons()
    inputs_train, inputs_validation, targets_train, targets_validation = split_data()
    
    # Build the model
    model = build_model(inputs_train.shape, num_classes)
    train_model(model,inputs_train,targets_train,inputs_validation,targets_validation)
    model.summary()
    model.save("src/model/cnn.h5")
    



def test_model():
    
    inputs_test, targets_test = load_data(json_path_train)
    model = tf.keras.saving.load_model("src/model/cnn.h5")
    evaluate_model(model,inputs_test,targets_test)

test_model() # Works fine via accuracy 0.92