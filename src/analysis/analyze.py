import librosa
import numpy as np

"""
 Analyze the speech signal
    First, extract mel frequency


"""

def setDirectory(dir):
    main(dir = dir)

def main(dir):
     # Load the audio file
    #audio_file = "../assets/s5.wav"
    audio_file = dir
    audio, sr = librosa.load(audio_file)

    print("Sampling Rate: " , sr)


    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    print("Mfcc:" , mfccs)

    # Transpose the MFCCs matrix to have the time axis first
    mfccs = np.transpose(mfccs)

    # one speaker, one audio file


    num_samples = mfccs.shape[0]
    #TODO: Fill target_labels with user speech
    target_labels = np.random.randint(0, 2, size=num_samples)

    print("target labels: ", target_labels)

    # Save the target labels array to a file
    np.save("target_labels.npy", target_labels)
    # Save the MFCCs matrix to a file
    np.save("example_mfccs.npy", mfccs)

if __name__ == "__main__":
    main()
   