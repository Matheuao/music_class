import math, random
import torch
import torchaudio
from torchaudio import transforms
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

from keras.preprocessing import image
import os

# AUDIO FILE MANIPULATION
def load_audio(file):
    audio_tensor, sr = torchaudio.load(file)
    
    return audio_tensor, sr

def save_audio(aud, path, sr):
    #sf.write(path, aud, samplerate=sr, 'PCM_24')

    torchaudio.save(path, aud, sample_rate=sr)
    
def rechannel(aud):
    
    # Convert from mono to stereo by duplicating the first channel
    resig = torch.cat([aud, aud])

    return ((resig))

def resample(aud, sr, newsr):
    sig = aud

    if (sr == newsr):
      # Nothing to do
      return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
        # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
        resig = torch.cat([resig, retwo])

    return ((resig, newsr))

def mp32wav(src, dst):
    subprocess.call(['ffmpeg', '-i', src, dst])
    
def crop_audio(aud, sr, time_frame):
    walk = int(random.uniform(0,aud.shape[1]))
    aud = aud[:,walk:walk+time_frame*sr]
              
    return aud

# AUDIO AUGMENTATION
def time_shift(aud, sr, shift_limit):
    audio_len = aud.shape
    shift_amt = round(random.random() * shift_limit * audio_len[1])
    aud_shifted = aud.roll(shift_amt), sr
    
    return aud_shifted

def pitch_shift(aud, sr):
    shift_amt = int(random.uniform(-1,1)*5+1)
    waveform = librosa.effects.pitch_shift(y=aud,sr=sr,n_steps=shift_amt)
    
    return waveform

# AUDIO SPECTROGRAM MANIPULATION AND AUGMENTATION
def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = torchaudio.load(audio_file)
    ms = transforms.MelSpectrogram(sr, n_fft=1024, hop_length=None, n_mels=64)(y)
    log_ms = transforms.AmplitudeToDB(top_db=80)(ms)
    librosa.display.specshow(torch.Tensor.numpy(log_ms[0]),sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
    return log_ms
    
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)
        
# SPECTROGRAM AUGMENTATION
def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

# FUNÇOES PARA GERAR ARRAY DE IMAGEM
# Pop - 0
# Metal - 1
# Disco - 2
# Blues - 3
# Reggae - 4 
# Classical - 5  
# Rock - 6 
# Hip-Hop - 7
# Country - 8
# Jazz - 9

def augmented_spec_loop(path_in, path_out):
   

    for folder in os.listdir(path_in):

        os.mkdir( path_out + "/" + folder)

        for file in os.listdir(path_in + "/" + folder):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            #../dataset/genres_augmented"+ "/" + folder + "/" + file
            audio_tensor, sr = load_audio(path_in + "/" + folder + "/" + file)
            ms = transforms.MelSpectrogram(sr, n_fft=1024, hop_length=None, n_mels=64)(audio_tensor)
            log_ms = transforms.AmplitudeToDB(top_db=80)(ms)
            spec_aug = spectro_augment(log_ms)
            librosa.display.specshow(torch.Tensor.numpy(spec_aug[0]),sr=sr)
            
            png = file.replace('.wav', '.png')

            fig.savefig(path_out + "/" + folder + "/" + png)
            plt.close(fig)

def load_images_from_path(path, label, resolution = [480,640]):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(int(resolution[0]), int(resolution[1]), 3))))
        labels.append((label))
        
    return images, labels

def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)