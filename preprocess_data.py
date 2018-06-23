import sys
import argparse
import librosa
import soundfile as sf
import numpy as np
import os
import csv
import pandas as pd
from constants import *
from encoding import make_encodings, encode


def preprocess_data(path, files_destination, labels_destination):
    letter2ind = make_encodings()
    os.makedirs(os.path.dirname(files_destination), exist_ok=True)
    
    wave_files = []
    encoded_labels = []

    reader_dirs = os.listdir(path)
    for reader_dir in reader_dirs:
        chapters_path = path + '/' + reader_dir
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
                    encoded_labels.append(encode(label, letter2ind))

    labels_df = pd.DataFrame(columns=['file', 'label'])
    files_num = len(wave_files)
    for i, (wave_file, label) in enumerate(zip(wave_files, encoded_labels)):
        wave_file_name = wave_file.split('/')[-1]
        mfcc_file_path = files_destination + wave_file_name.split('.')[0] + '.npy'
        
        print('{}/{}\t{}'.format(i, files_num, wave_file_name))
        wave, sr = sf.read(wave_file)
        mfcc = librosa.feature.mfcc(wave, sr=sr)
        # save mfcc
        np.save(mfcc_file_path, mfcc, allow_pickle=False)

        # save filename and encoded label
        labels_df.loc[i] = [wave_file_name, label]
    labels_df.to_csv(labels_destination, sep=',')
 

def run_preprocessing():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', required = False, help='Preprocessing test set')
    ap.add_argument('--training', required = False, help='Preprocessing training set')
    args = vars(ap.parse_args())
    if args['test']:
        path = DATASET_DESTINATION + TEST
        files_destination = DATASET_DESTINATION + PREPROCESSED_TEST + '/'
        labels_destination = DATASET_DESTINATION + LABELS_TEST
        preprocess_data(path, files_destination, labels_destination)
    if args['training']:
        path = DATASET_DESTINATION + TRAIN
        files_destination = DATASET_DESTINATION + PREPROCESSED_TRAIN + '/'
        labels_destination = DATASET_DESTINATION + LABELS_TRAIN
        preprocess_data(path, files_destination, labels_destination)


if __name__ == '__main__':
    run_preprocessing()