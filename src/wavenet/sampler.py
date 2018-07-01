import torch
import numpy as np
from torch.utils.data.sampler import Sampler


class DatasetSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        self.data_source = data_source
        samples = list(range(0, len(data_source)))
        self.batches = [samples[i: i + batch_size] for i in range(0, len(samples), batch_size)]
    
    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)
