import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def wav_to_spec(directory, save_dir, save_array_dir):
    counter = 0
    # dir = os.fsencode(directory)
    for file in os.scandir(directory):
        # loading wav file with desired sampling rate
        wav, sr = librosa.load(file, sr=24000) # sr in LibriTTS = 24kHz
        # applying stft
        stft_wav = librosa.stft(wav,n_fft=1024, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
        spec = librosa.amplitude_to_db(abs(stft_wav), ref=np.max)
        # extracting mel spectrogram with FastPitch regulations 
        mel_spec = librosa.feature.melspectrogram(S=spec, n_fft=2048, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0)
        librosa.display.specshow(mel_spec, sr=sr)
        
        np.save(f"{save_array_dir}/fastpitch_array{counter:05}", mel_spec)
        plt.savefig(f"{save_dir}/fastpitch_mel_spectrogram{counter:05}.png")
        
        counter += 1

def make_spec_from_array(array_dir, new_spec_dir, sr):
    for array in os.scandir(array_dir):
        if array.is_file() and array.name.endswith('.npy'):
            spec_array = np.load(array.path)
        
        librosa.display.specshow(data=spec_array, sr=sr)

        # Construct the output file name
        output_file_name = f"spec_{array.name}.png"
        print("output_file_name: ", output_file_name)
        output_file_path = os.path.join(new_spec_dir, output_file_name)
        print("output_file_path: ", output_file_path)
        plt.savefig(f"{output_file_path}")


def chop_arrays(target_size_x, target_size_y, save_array_dir, chop_array_dir):
    
    for array in os.scandir(save_array_dir):
        if array.is_file() and array.name.endswith('.npy'):
            # Load the array from the file
            spec_array = np.load(array.path)
            rows, cols = spec_array.shape

            if rows < target_size_x or cols < target_size_y:
                raise ValueError("Array too small for desired chop")
            
            # centering the array with starting indices
            start_row = (rows - target_size_x)//2
            start_col = (cols - target_size_y)//2

            # centering array with end indices
            end_row = start_row + target_size_x
            end_col =  start_col + target_size_y

            # slicing array for center wav
            chopped_array = spec_array[start_row:end_row, start_col:end_col]

            # Construct the output file name
            output_file_name = f"80x80_{array.name}"
            print("output_file_name: ", output_file_name)
            output_file_path = os.path.join(chop_array_dir, output_file_name)
            print("output_file_path: ", output_file_path)

            # Save the chopped array
            np.save(output_file_path, chopped_array)


current_dir = "/Users/marie/Desktop/autoencoder/example_wavs"
save_dir = "./saved_spectrograms"
save_array_dir = "./spec_arrays"
save_chop = "./chopped_arrays"
save_chop_spec = "./chopped_specs"
# wav_to_spec(current_dir, save_dir, save_array_dir)
# chop_arrays(target_size_x=80, target_size_y=80, save_array_dir=save_array_dir, chop_array_dir=save_chop)
make_spec_from_array(save_chop, save_chop_spec, 24000)
