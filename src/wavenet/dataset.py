import torch
import pandas as pd
import numpy as np
from definitions import *
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, labels_path, files_path):
        self.mfcc_files = []
        self.labels = []
        data = pd.read_csv(labels_path, sep='\t')
        for index, row in data.iterrows():
            file_path = files_path + row['file'].replace('flac', 'npy')
            self.mfcc_files.append(file_path)
            self.labels.append(eval(row['label']))
        
    def __len__(self):
        return len(self.mfcc_files)
    
    def __getitem__(self, index):
        label = self.labels[index]
        file_path = self.mfcc_files[index]
        
        mfcc = np.load(file_path, allow_pickle=False)
        mfcc_tensor = torch.FloatTensor(mfcc)

        return mfcc_tensor, label


if __name__ == '__main__':
    sd = SpeechDataset(DATASET_DESTINATION + LABELS_TEST, DATASET_DESTINATION + PREPROCESSED_TEST)
    print(sd.__getitem__(3))
