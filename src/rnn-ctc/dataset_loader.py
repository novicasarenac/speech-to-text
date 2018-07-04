import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from src.definitions import LABELS_TRAIN
from src.definitions import PREPROCESSED_RNN_TRAIN
from src.definitions import DATASET_DESTINATION


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

    def __getitem__(self, index):
        labels = torch.from_numpy(self.labels[index])
        mfcc = torch.from_numpy(np.load(self.features[index],
                                        allow_pickle=False))
                    .unsqueeze_(1)

        return mfcc, labels


if __name__ == "__main__":
    labels_path = DATASET_DESTINATION + LABELS_TRAIN
    features_path = DATASET_DESTINATION + PREPROCESSED_RNN_TRAIN
    loader = DatasetLoader(labels_path, features_path)

    mfcc, label = loader[3]
    print(mfcc.shape)
