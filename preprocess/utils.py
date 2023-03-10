import os
import librosa
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor

default = True  # True代表默认22050Hz False代表原始16000Hz


def get_wavs(base_path: str = "EmoDB"):
    wav_files = []
    for dir in os.listdir(base_path):
        for file in os.listdir(base_path + "/" + dir):
            wav_files.append(base_path + "/" + dir + "/" + file)
    return wav_files


def get_mfcc(filename, duration=4, framelength=0.05):
    if default:
        data, sr = librosa.load(filename)
    else:
        data, sr = librosa.load(filename, sr=None)
    time = librosa.get_duration(y=data, sr=sr)
    if time > duration:
        data = data[0:int(sr * duration)]
    else:
        padding_len = int(sr * duration - len(data))
        data = np.hstack([data, np.zeros(padding_len)])
    framesize = int(framelength * sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=framesize)
    mfcc = mfcc.T
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    mfcc_acc = librosa.feature.delta(mfcc_delta, width=3)
    mfcc = np.hstack([mfcc, mfcc_delta, mfcc_acc])
    return mfcc


# MFCC(13)
def get_mfccs(wav_files: list, duration=4, framelength=0.05):
    print("get mfcc")
    mfccs = get_mfcc(wav_files[0], duration=duration, framelength=framelength)
    size = mfccs.shape
    for it in tqdm(wav_files[1:]):
        mfcc = get_mfcc(it, duration=duration, framelength=framelength)
        mfccs = np.vstack((mfccs, mfcc))
    mfccs = mfccs.reshape(-1, size[0], size[1])
    # reshape(-1, size[1], size[0])与 reshape(-1, size[0], size[1]).transpose([0,2,1])的结果不同
    return mfccs


def get_melspectrum(filename, duration=4, framelength=0.05, n_mels=40):
    if default:
        data, sr = librosa.load(filename)
    else:
        data, sr = librosa.load(filename, sr=None)
    time = librosa.get_duration(y=data, sr=sr)
    if time > duration:
        data = data[0:sr * duration]
    else:
        padding_len = sr * duration - len(data)
        data = np.hstack([data, np.zeros(padding_len)])
    framesize = int(framelength * sr)
    mel_spect = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=framesize, n_mels=n_mels)  # n_mels 维
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect


# 获取mel声谱图（fbank）
def get_spects(wav_files: list, duration=4, framelength=0.05, n_mels=40):
    print("mel spectrum")
    mel_spects = get_melspectrum(wav_files[0], duration, framelength, n_mels=n_mels)
    size = mel_spects.shape
    for it in tqdm(wav_files[1:]):
        mel_spect = get_melspectrum(it, duration, framelength)
        mel_spects = np.vstack((mel_spects, mel_spect))
    mel_spects = mel_spects.reshape(-1, size[0], size[1])
    return mel_spects


def get_mask(mfcc_len, mfcc_base):
    if mfcc_base >= mfcc_len:
        mask = np.hstack((np.ones([1, mfcc_len]), np.zeros([1, mfcc_base - mfcc_len])))
    else:
        mask = np.ones([1, mfcc_base])
    return mask


def get_masks(wav_files: list, duration=4, framelength=0.05):
    print("get mask")
    filename = wav_files[0]
    if default:
        data, sr = librosa.load(filename)
    else:
        data, sr = librosa.load(filename, sr=None)
    time = librosa.get_duration(y=data, sr=sr)
    framesize = int(framelength * sr)
    if time > duration:
        datapad = data[0:int(sr * duration)]
    else:
        padding_len = int(sr * duration - len(data))
        datapad = np.hstack([data, np.zeros(padding_len)])
    mfcc_base = librosa.feature.mfcc(y=datapad, sr=sr, n_mfcc=13, n_fft=framesize).shape[1]
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=framesize)
    mfcc_len = mfcc.shape[1]
    masks = get_mask(mfcc_len, mfcc_base)
    for it in tqdm(wav_files[1:]):
        if default:
            data, sr = librosa.load(it)
        else:
            data, sr = librosa.load(it, sr=None)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13, n_fft=framesize)
        mfcc_len = mfcc.shape[1]
        mask = get_mask(mfcc_len, mfcc_base)
        masks = np.vstack([masks, mask])
    return masks


# one-hot编码
def class_code(class_name):
    if class_name == 'anger':
        return np.array([1, 0, 0, 0, 0, 0, 0])
    elif class_name == 'boredom':
        return np.array([0, 1, 0, 0, 0, 0, 0])
    elif class_name == 'disgust':
        return np.array([0, 0, 1, 0, 0, 0, 0])
    elif class_name == 'fear':
        return np.array([0, 0, 0, 1, 0, 0, 0])
    elif class_name == 'happy':
        return np.array([0, 0, 0, 0, 1, 0, 0])
    elif class_name == 'neutral':
        return np.array([0, 0, 0, 0, 0, 1, 0])
    elif class_name == 'sad':
        return np.array([0, 0, 0, 0, 0, 0, 1])
    else:
        return np.zeros([1, 7])


# 获取标签
def get_labels(wav_files):
    y = None
    for it in wav_files:
        if y is None:
            y = class_code(it.split('/')[-2])
        else:
            y = np.vstack((y, class_code(it.split('/')[-2])))
    return y


def load_wavs(base_path: str = "./EmoDB", duration:float = 4):
    wav_files = get_wavs(base_path=base_path)
    # labels = get_labels(wav_files=wav_files)
    wav_data = None
    for it in tqdm(wav_files):
        data, sr = librosa.load(it, sr=None)
        time = librosa.get_duration(y=data, sr=sr)
        if time > duration:
            datapad = data[0:int(sr * duration)]
        else:
            padding_len = int(sr * duration - len(data))
            datapad = np.hstack([data, np.zeros(padding_len)])

        processor = Wav2Vec2Processor.from_pretrained("../pretrained")
        data = processor(datapad, return_tensors="pt", padding="longest", sampling_rate=sr).input_values
        if wav_data is None:
            wav_data = data
        else:
            wav_data = np.vstack([wav_data, data])
    np.save("x_wav.npy", wav_data)
    return wav_data.shape
    # print()


# 加载数据集
def load_dataset(base_path: str, is_save=True, duration=4, framelength=0.05):
    if os.path.exists("./x_mfcc.npy") and os.path.exists("./y.npy"):
        print("exist x exist y")
        x = np.load("x_mfcc.npy")
        y = np.load("y.npy")
    else:
        wav_files = get_wavs(base_path=base_path)
        x = get_mfccs(wav_files=wav_files, duration=duration, framelength=framelength)
        y = get_labels(wav_files=wav_files)
        x = np.array(x, dtype='float32')
        y = np.array(y, dtype='float32')
        if is_save:
            np.save("x_mfcc.npy", x)
            np.save("y.npy", y)
    return x, y
