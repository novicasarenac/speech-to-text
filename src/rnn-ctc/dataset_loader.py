import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    def __init__(self, labels_path, features_path):
        self.labels = []
        self.features = []

        labels_file = pd.read_csv(labels_path, sep='\t')
        for _, row in labels_file.iterrows():
            self.labels.append(np.array(eval(row['label'])))
            self.features.append(features_path + row['file'].replace('flac', 'npy'))

    def __len__(self):
        return len(self.features)

    def max_label_length(self):
        return max(list(map(lambda label: len(label), self.labels)))

    def __getitem__(self, index):
        label = self.labels[index]

        labels = torch.from_numpy(self._pad_label_sequence(label, self.max_label_length()))
        mfcc = torch.from_numpy(np.load(self.features[index], allow_pickle=False)).unsqueeze_(1)

        return mfcc, labels

    def _pad_label_sequence(self, label, max_size):
        if len(label) < max_size:
            return np.pad(label, (0, max_size - len(label)), mode='constant')
        else:
            return label
