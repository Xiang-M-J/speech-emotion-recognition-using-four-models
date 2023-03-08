import random

import numpy as np
import soundfile as sf
from tqdm import tqdm

from augment import noise, dyn_change, stretch, shift, augment_apply, augment_torch
from utils import get_wavs, get_masks, get_spects, get_mfccs, get_labels, load_wavs


def mask_process(base_path="EmoDB", duration=4, framelength=0.05, save_name: str = "mask.npy"):
    wav_files = get_wavs(base_path=base_path)
    masks = get_masks(wav_files, duration=duration, framelength=framelength)
    np.save(save_name, masks)
    print(masks.shape)


def mfcc_process(base_path="EmoDB", duration=4, framelength=0.05, save_name: list = None):
    if save_name is None:
        save_name = ["x_mfcc.npy", "y.npy"]
    wav_files = get_wavs(base_path=base_path)
    x_mfcc = get_mfccs(wav_files, duration=duration, framelength=framelength)
    y = get_labels(wav_files=wav_files)
    x_mfcc = np.array(x_mfcc, dtype="float32")
    y = np.array(y, dtype="float32")
    np.save(save_name[0], x_mfcc)
    np.save(save_name[1], y)


def mel_process(base_path="EmoDB", duration=4, framelength=0.05, save_name: str = "x_mel.npy"):
    wav_files = get_wavs(base_path=base_path)
    x_mel = get_spects(wav_files, duration=duration, framelength=framelength)
    x_mel = np.array(x_mel, dtype="float32")
    np.save(save_name, x_mel)
    print(x_mel.shape)


def audio_augment():
    files_index = {
        'anger': [0, 126], 'boredom': [127, 207], 'disgust': [208, 253], 'fear': [254, 322], 'happy': [323, 393],
        'neutral': [394, 472], 'sad': [473, 534]
    }
    methods = [noise, dyn_change, stretch, shift]
    wav_files = get_wavs("EmoDB")
    files = [wav_files[files_index['boredom'][0]:files_index['boredom'][1] + 1],
             wav_files[files_index['disgust'][0]:files_index['disgust'][1] + 1],
             wav_files[files_index['fear'][0]:files_index['fear'][1] + 1],
             wav_files[files_index['happy'][0]:files_index['happy'][1] + 1],
             wav_files[files_index['neutral'][0]:files_index['neutral'][1] + 1],
             wav_files[files_index['sad'][0]:files_index['sad'][1] + 1]]
    augment_num = [39, 34, 31, 49, 31, 31]
    augment_files = []
    for i in range(len(files)):
        augment_files.extend(random.sample(files[i], augment_num[i]))
    augment_method = {}

    for it in augment_files:
        augment_method[it] = np.random.choice(methods, 1)
    print(len(augment_method))
    for key, value in tqdm(augment_method.items()):
        data_, sr, new_name = augment_apply(key, value)
        sf.write(new_name, data_, sr)


if __name__ == "__main__":
    # mel_process(base_path="EmoDB", duration=4, framelength=0.05)
    # audio_augment()
    # mfcc_process(base_path="EmoDBPro", duration=4, framelength=0.05, save_name=["x_mfcc_a.npy", "y_a.npy"])
    # mel_process(base_path="EmoDBPro", duration=4, framelength=0.05, save_name="x_mel_a.npy")
    # mask_process(base_path="EmoDBPro", duration=4, framelength=0.05, save_name="mask_a.npy")
    # wav_data, label = load_wavs()
    # np.save("x_vec.npy", wav_data)
    # print(load_wavs())  # output [535, 64000]
    effect = ['dither']
    wav_file = 'EmoDB/anger/10a01Wa.wav'
    data = augment_torch(wav_file, effect)
    pass
