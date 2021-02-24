from os import path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class InferenceDataset(Dataset):

    def __init__(self,dataset):
        super().__init__()
        self.dataset = dataset
        assert self.dataset in ['iper','h36m']

