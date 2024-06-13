import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def wav_to_spec(directory, save_dir):
    counter = 0
    # dir = os.fsencode(directory)
    for file in os.scandir(directory):
        y, sr = librosa.load(file, sr=None) # sr in LibriTTS = 24kHz
        print("sr ", sr)
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr)
        # plt.title('Waveform')
        # plt.savefig(f"{save_dir}/spectrogram{counter:05}.png")
        counter += 1



current_dir = "/Users/marie/Desktop/autoencoder/example_wavs"
save_dir = "./saved_spectrograms"
wav_to_spec(current_dir, save_dir)

# y, sr = librosa.load("./example_wavs/sent1.wav", sr=None)
# D = librosa.stft(y)
# S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
# librosa.display.specshow(S_db, sr=sr)
# plt.figure(figsize=(10, 6))
# librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (dB)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# # plt.show()
# plt.savefig("spectrogram3")
# # plt.title('Waveform')
# plt.savefig(f"{save_dir}/spectrogram{counter:05}.png")