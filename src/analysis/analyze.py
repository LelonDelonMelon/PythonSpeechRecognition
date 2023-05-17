import librosa
import numpy as np

"""
 Analyze the speech signal
    First, extract mel frequency
"""

# Load the audio file
audio_file = "assets/s5.wav"
audio, sr = librosa.load(audio_file, sr=None)

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# Transpose the MFCCs matrix to have the time axis first
mfccs = np.transpose(mfccs)

# one speaker, one audio file
target_labels = np.zeros([47])

# Save the target labels array to a file
np.save("target_labels.npy", target_labels)
# Save the MFCCs matrix to a file
np.save("example_mfccs.npy", mfccs)
