import os
import sys
import argparse
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from encoding import make_encodings, encode
from python_speech_features import base
from definitions import *


def preprocess_data(data_path, files_destination, labels_destination, mfcc_type):
    wave_files, encoded_labels = read_data_files(data_path, files_destination)
    extract_mfcc(wave_files,
                 encoded_labels,
                 files_destination,
                 labels_destination,
                 mfcc_type)


def read_data_files(data_path, files_destination):
    letter_indices = make_encodings()
    os.makedirs(os.path.dirname(files_destination), exist_ok=True)

    wave_files = []
    encoded_labels = []

    reader_dirs = os.listdir(data_path)
    for reader_dir in reader_dirs:
        chapters_path = data_path + '/' + reader_dir
        chapter_dirs = os.listdir(chapters_path)
        for chapter_dir in chapter_dirs:
            speech_dir = chapters_path + '/' + chapter_dir
            files = os.listdir(speech_dir)

            meta_file = list(filter(lambda f: f.endswith('.txt'), files))[0]
            meta_file_path = speech_dir + '/' + meta_file
            with open(meta_file_path, 'r') as metafile:
                for line in metafile:
                    content = line.strip()
                    file, label = content.split(' ', 1)
                    file_path = speech_dir + '/' + file + '.flac'

                    wave_files.append(file_path)
                    encoded_labels.append(encode(label, letter_indices))

    return wave_files, encoded_labels


def extract_mfcc(wave_files, encoded_labels, files_destination, labels_destination, mfcc_type):
    labels_df = pd.DataFrame(columns=['file', 'label'])
    files_num = len(wave_files)

    for i, (wave_file, label) in enumerate(zip(wave_files, encoded_labels)):
        wave_file_name = wave_file.split('/')[-1]
        mfcc_file_path = files_destination + wave_file_name.split('.')[0] + '.npy'

        print('{}/{}\t{}'.format(i + 1, files_num, wave_file_name))
        wave_data, sample_rate = sf.read(wave_file)
        # save mfcc
        if mfcc_type == 'cnn':
            mfcc = librosa.feature.mfcc(wave_data, sr=sample_rate)
        elif mfcc_type == 'rnn':
            mfcc = base.mfcc(wave_data,
                             samplerate=sample_rate,
                             numcep=13,
                             winstep=0.01,
                             winfunc=np.hamming)
            deltas = base.delta(mfcc, 2)

            # normalize mfcc over all frames
            mfcc_mean = np.mean(mfcc, axis=0)
            mfcc_std = np.std(mfcc, axis=0)
            mfcc = (mfcc - mfcc_mean)/mfcc_std

            # normalize deltas over all frames
            delta_mean = np.mean(deltas, axis=0)
            delta_std = np.std(deltas, axis=0)
            deltas = (deltas - delta_mean)/delta_std

        np.save(mfcc_file_path,
                np.concatenate((mfcc, deltas), axis=1),
                allow_pickle=False)

        labels_df.loc[i] = [wave_file_name, label]

    labels_df.to_csv(labels_destination,
                     sep='\t',
                     index=False)


def run_preprocessing():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test',
                    nargs=1,
                    type=str,
                    required=False,
                    help='Preprocessing test set. Params: \'cnn\' | \'rnn\'')

    ap.add_argument('--training',
                    nargs=1,
                    type=str,
                    required=False,
                    help='Preprocessing training set. Params: \'cnn\' | \'rnn\'')

    args = ap.parse_args()
    if args.test is None and args.training is None:
        print("Unknown argument. Check help for valid options.")
        sys.exit(1)

    if args.test:
        mfcc_type = args.test[0]
        if args.test[0] == 'cnn':
            feature_destination = DATASET_DESTINATION + PREPROCESSED_CNN_TEST
        elif args.test[0] == 'rnn':
            feature_destination = DATASET_DESTINATION + PREPROCESSED_RNN_TEST
        else:
            print('Unknown parameter. Exiting...')
            sys.exit(1)

        data_path = DATASET_DESTINATION + TEST
        labels_destination = DATASET_DESTINATION + LABELS_TEST
    if args.training:
        mfcc_type = args.training[0]
        if args.training[0] == 'cnn':
            feature_destination = DATASET_DESTINATION + PREPROCESSED_CNN_TRAIN
        elif args.training[0] == 'rnn':
            feature_destination = DATASET_DESTINATION + PREPROCESSED_RNN_TRAIN
        else:
            print('Unknown parameter. Exiting...')
            sys.exit(1)

        data_path = DATASET_DESTINATION + TRAIN
        labels_destination = DATASET_DESTINATION + LABELS_TRAIN

    preprocess_data(data_path,
                    feature_destination,
                    labels_destination,
                    mfcc_type)


if __name__ == '__main__':
    run_preprocessing()
