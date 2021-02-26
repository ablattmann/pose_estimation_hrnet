import os
from os import path
import pickle
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import cv2


class InferenceDataset(Dataset):

    def __init__(self,dataset):
        super().__init__()
        self.dataset = dataset
        self.transforms = T.Compose([T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        assert self.dataset in ['iper','h36m','taichi']

        if self.dataset == 'iper':
            self.metafile = '/export/scratch/compvis/datasets/iPER/processed_256_resized/iper_256_resized_frange.p'
            self.datapath = '/export/scratch/compvis/datasets/iPER/processed_256_resized'
        elif self.dataset == 'taichi':
            self.metafile = '/export/scratch/compvis/datasets/taichi/taichi/meta_with_10_20.p'
            self.datapath = '/export/scratch/compvis/datasets/taichi/taichi/'
        else:
            self.metafile = "/export/scratch/compvis/datasets/human3.6M/video_prediction/h36m_test_smaller.p"
            self.datapath = '/export/scratch/compvis/datasets/human3.6M/video_prediction'

        if 'DATAPATH' in os.environ:
            self.metafile = path.join(os.environ['DATAPATH'],self.metafile[1:])
            self.datapath = path.join(os.environ['DATAPATH'],self.datapath[1:])
        with open(self.metafile,'rb') as f:
            self.data = pickle.load(f)

        self.data["img_path"] = [
            path.join(self.datapath, p if not p.startswith("/") else p[1:]) for p in self.data["img_path"]
        ]

        self.data = {key: np.asarray(self.data[key]) for key in self.data}

        self.data.update({'did':np.arange(self.data['img_path'].shape[0])})




    def __len__(self):
        return self.data['img_path'].shape[0]

    def __getitem__(self, id):

        return {'id' : self._get_id(id),
                'img' : self._get_img(id)}

    def _get_id(self,idx):
        return self.data['did'][idx]

    def _get_img(self,idx):

        img_path = self.data['img_path'][idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,dsize=(256,256),interpolation=cv2.INTER_LINEAR)


        img = self.transforms(img)
        return img
