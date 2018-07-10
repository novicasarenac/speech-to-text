import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from wavenet.dataset import SpeechDataset
from wavenet.sampler import DatasetSampler


class SpeechDataLoader(DataLoader):
    def __init__(self, labels_path, files_path, batch_size):
        dataset = SpeechDataset(labels_path, files_path)
        sampler = DatasetSampler(dataset, batch_size=batch_size)

        super(SpeechDataLoader, self).__init__(dataset, batch_sampler=sampler)
        self.collate_fn = self._colate_fn

    def _colate_fn(self, batch):
        longest_sample = max(batch, key=lambda sample: sample[0].size(1))[0]
        max_sample_length = longest_sample.size(1)
        freq_size = longest_sample.size(0)
        batch_size = len(batch)
        
        inputs = torch.zeros(batch_size, freq_size, max_sample_length)
        input_percentages = torch.FloatTensor(batch_size)
        target_sizes = torch.IntTensor(batch_size)
        targets = []
        for i in range(batch_size):
            sample = batch[i]
            tensor = sample[0]
            transcription = sample[1]
            seq_length = tensor.size(1)
            inputs[i].narrow(1, 0, seq_length).copy_(tensor)
            input_percentages[i] = seq_length / float(max_sample_length)
            target_sizes[i] = len(transcription)
            targets.extend(transcription)
        targets = torch.IntTensor(targets)
        return inputs, targets, input_percentages, target_sizes
        