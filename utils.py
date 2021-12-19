# USING LIBROSA IN THIS MODULE CUZ TENSORFLOW AUDIO IS SOMEHOW UNABLE TO READ A WAV FILE HOW IS THAT EVEN POSSIBLE WHY IS LIBROSA ABLE TO DO IT WIHOUT ERRORS BUT TENSORFLOW DOESNT JUST READ THE FILE ??

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import librosa
import numpy as np

EMOTION_DICT_RAVDEES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Data augmentations not implemented yet


def load_wav(file_path):
    file_path = file_path.numpy()
    wav, sr = librosa.load(file_path, mono=True, duration=3)

    pre_emp = 0.97
    wav = np.append(wav[0], wav[1:] - pre_emp * wav[:-1])

    wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    sr = tf.convert_to_tensor(sr, dtype=tf.float32)
    return wav, sr


def get_framed_mel_spectrograms(wav, sr=22050):
    # The duration of clips is 3 seconds, ie. 3000 miliseconds. Do some quick math to figure out frame_length.
    frame_length = tf.cast(sr * (25 / 1000), tf.int32)  # 25 ms
    frame_step = tf.cast(sr * (10 / 1000), tf.int32)  # 10 ms
    stft_out = tf.signal.stft(
        wav,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=tf.signal.hamming_window,
    )
    num_spectrogram_bins = tf.shape(stft_out)[-1]
    stft_abs = tf.abs(stft_out)
    lower_edge_hz, upper_edge_hz = 20.0, 8000.0
    num_mel_bins = 64
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sr, lower_edge_hz, upper_edge_hz
    )
    mel_spectrograms = tf.tensordot(stft_abs, linear_to_mel_weight_matrix, 1)

    # mel_spectrograms.set_shape(
    #     stft_abs.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    # )

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    log_mel_d1 = log_mel_spectrograms - tf.roll(log_mel_spectrograms, -1, axis=0)
    log_mel_d2 = log_mel_d1 - tf.roll(log_mel_d1, -1, axis=0)

    log_mel_three_channel = tf.stack(
        [log_mel_spectrograms, log_mel_d1, log_mel_d2], axis=-1
    )

    framed_log_mels = tf.signal.frame(
        log_mel_three_channel, frame_length=64, frame_step=32, pad_end=False, axis=0
    )

    return framed_log_mels


def get_dataset(DATA_DIR: str, cache: bool = True):
    def decompose_label(file_path: str):
        return label_to_int[file_path.split("-")[2]]

    def tf_compatible_file_loader(file_path):
        wav, sr = tf.py_function(load_wav, [file_path], [tf.float32, tf.float32])
        return wav, sr

    file_path_list = os.listdir(DATA_DIR)
    label_to_int = dict({(key, i) for i, key in enumerate(EMOTION_DICT_RAVDEES.keys())})

    labels = [decompose_label(file_path) for file_path in file_path_list]
    file_path_list = [DATA_DIR + "/" + file_path for file_path in file_path_list]
    train_fps, val_fps, train_labels, val_labels = train_test_split(
        file_path_list, labels, test_size=0.1
    )

    train_files_ds = tf.data.Dataset.from_tensor_slices(train_fps)
    train_wav_ds = train_files_ds.map(
        tf_compatible_file_loader,  num_parallel_calls=tf.data.AUTOTUNE
    )
    train_mfcc_ds = train_wav_ds.map(
        get_framed_mel_spectrograms,  num_parallel_calls=tf.data.AUTOTUNE
    )

    train_labels_ds = tf.data.Dataset.from_tensor_slices(train_labels)

    train_ds = tf.data.Dataset.zip((train_mfcc_ds, train_labels_ds))

    val_files_ds = tf.data.Dataset.from_tensor_slices(val_fps)
    val_wav_ds = val_files_ds.map(
        tf_compatible_file_loader,  num_parallel_calls=tf.data.AUTOTUNE
    )
    val_mfcc_ds = val_wav_ds.map(
        get_framed_mel_spectrograms,  num_parallel_calls=tf.data.AUTOTUNE
    )

    val_labels_ds = tf.data.Dataset.from_tensor_slices(val_labels)

    val_ds = tf.data.Dataset.zip((val_mfcc_ds, val_labels_ds))

    if cache:
        train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE).cache()
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE).cache()
    else:
        train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds
