import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import wave

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

# Data augmentations ka bhi add karo yahan 

def load_wav(file_path):
    # https://github.com/mozilla/DeepSpeech/issues/2048
    import pandas
    import sys

    def compare_header_and_size(wav_filename):
        with wave.open(wav_filename, 'r') as fin:
            header_fsize = (fin.getnframes() * fin.getnchannels() * fin.getsampwidth()) + 44
        file_fsize = os.path.getsize(wav_filename)
        return header_fsize != file_fsize

    df = pandas.read_csv(sys.argv[1])
    invalid = df.apply(lambda x: compare_header_and_size(x['wav_filename']), axis=1)
    print('The following files are corrupted:')
    print(df[invalid].values)

    pre_emp = 0.97
    file_contents = tf.io.read_file(file_path)

    # Default SR is 22050, putting desired_samples to 66150 to get 3 second sample
    wav, sr = tf.audio.decode_wav(
        file_contents, desired_channels=1, desired_samples=66150
    )

    wav = tf.squeeze(wav, axis=-1)

    # Apply preamp if needed
    wav = tf.experimental.numpy.append(wav[0], wav[1:] - pre_emp * wav[:-1])

    return wav, sr


def get_mfcc(wav, sr=22050):
    stft_out = tf.signal.stft(
        wav, 400, 160, window_fn=tf.signal.hamming_window, pad_end=False
    )
    num_spectrogram_bins = tf.shape(stft_out)[-1]
    stft_abs = tf.abs(stft_out)
    lower_edge_hz, upper_edge_hz = 20.0, 8000.0
    num_mel_bins = 64
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sr, lower_edge_hz, upper_edge_hz
    )
    mel_spectrograms = tf.tensordot(stft_abs, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        stft_abs.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )

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

def get_dataset(
    DATA_DIR: str,
):
    def decompose_label(file_path: str):
        return label_to_int[file_path.split("-")[2]]

    file_path_list = os.listdir(DATA_DIR)
    label_to_int = dict({(key, i) for i, key in enumerate(EMOTION_DICT_RAVDEES.keys())})

    labels = [decompose_label(file_path) for file_path in file_path_list]
    train_fps, val_fps, train_labels, val_labels = train_test_split(
        file_path_list, labels, test_size=0.1
    )

    train_files_ds = tf.data.Dataset.from_tensor_slices(train_fps)
    train_wav_ds = train_files_ds.map(load_wav, num_parallel_calls=tf.data.AUTOTUNE)
    train_mfcc_ds = train_wav_ds.map(get_mfcc, num_parallel_calls=tf.data.AUTOTUNE)

    train_labels_ds = tf.data.Dataset.from_tensor_slices(train_labels)

    train_ds = tf.data.Dataset.zip((train_mfcc_ds, train_labels_ds))

    val_files_ds = tf.data.Dataset.from_tensor_slices(val_fps)
    val_wav_ds = val_files_ds.map(load_wav, num_parallel_calls=tf.data.AUTOTUNE)
    val_mfcc_ds = val_wav_ds.map(get_mfcc, num_parallel_calls=tf.data.AUTOTUNE)

    val_labels_ds = tf.data.Dataset.from_tensor_slices(val_labels)

    val_ds = tf.data.Dataset.zip((val_mfcc_ds, val_labels_ds))

    train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds