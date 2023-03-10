#########################
# Augmentation methods
#########################
import numpy as np
import librosa
import torchaudio
import random


def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.05 * np.random.uniform() * np.amax(data)  # more noise reduce the value to 0.5
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data, "noise"


def shift(data):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high=5) * 1000)  # default at 500
    return np.roll(data, s_range), "shift"


def stretch(data):
    """
    Stretching the Sound. Note that this expands the dataset slightly
    """
    rate = np.random.uniform(low=0.8, high=1.2)
    data = librosa.effects.time_stretch(y=data, rate=rate)
    return data, "shift"


def pitch(data, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    data = librosa.effects.pitch_shift(y=data.astype('float64'),
                                       sr=sample_rate, n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)
    return data, 'pitch'


def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=1.5, high=2)  # default low = 1.5, high = 3
    return data * dyn_change, "dyn_change"


def speedNpitch(data):
    """
    speed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.2 / length_change  # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0, len(data), speed_fac), np.arange(0, len(data)), data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data, 'speedNpitch'


def augment_apply(file_name, method, new_dir="EmoDBPro"):
    data, sr = librosa.load(file_name, sr=None)
    data_, method_name = method[0](data)
    new_name = str(file_name).split('.')[0] + f"_{method_name}" + ".wav"
    new_name = new_name.replace("EmoDB", new_dir)
    return data_, sr, new_name


def augment_torch(filename, effect):  # 暂时无法使用
    wave, sr = torchaudio.load(filename)
    wave_a, sr = torchaudio.sox_effects.apply_effects_tensor(wave, sr, effect)
    return wave_a
