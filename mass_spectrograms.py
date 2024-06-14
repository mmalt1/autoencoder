import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def wav_to_spec(directory, save_dir, save_array_dir):
    counter = 0
    # dir = os.fsencode(directory)
    for file in os.scandir(directory):
    #     y, sr = librosa.load(file, sr=None) # sr in LibriTTS = 24kHz
    #     print("sr ", sr)
    #     D = librosa.stft(y)
    #     S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
    #     librosa.display.specshow(S_db, sr=sr)
    #     # plt.title('Waveform')
    #     # plt.savefig(f"{save_dir}/spectrogram{counter:05}.png")
    #     counter += 1
        wav, sr = librosa.load(file, sr=24000) # sr in LibriTTS = 24kHz
        print("sr ", sr)
        stft_wav = librosa.stft(wav,n_fft=1024, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
        spec = librosa.amplitude_to_db(abs(stft_wav), ref=np.max)
        mel_spec = librosa.feature.melspectrogram(S=spec)
        librosa.feature.melspectrogram(S=spec, n_fft=2048, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0)
        librosa.display.specshow(mel_spec, sr=sr)
        np.save(f"{save_array_dir}/fastpitch_array{counter:05}", mel_spec)
        print(mel_spec)
        print(mel_spec.size)
        print(mel_spec.shape)
        # plt.title('Waveform')
        plt.savefig(f"{save_dir}/fastpitch_mel_spectrogram{counter:05}.png")
        counter += 1



current_dir = "/Users/marie/Desktop/autoencoder/example_wavs"
save_dir = "./saved_spectrograms"
save_array_dir = "./spec_arrays"
wav_to_spec(current_dir, save_dir, save_array_dir)

