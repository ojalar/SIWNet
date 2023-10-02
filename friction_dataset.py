from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torch


class FrictionDataset(Dataset):
    # implementation of the dataset used in the study.
    def __init__(self, data_paths, transform):
        # load data from csv-files
        data_list = [pd.read_csv(path, header=None) for path in data_paths]
        self.data = pd.concat(data_list)
        self.data_len = len(self.data.index)
        self.img_path = np.asarray(self.data.iloc[:, 0])
        self.grip = np.asarray(self.data.iloc[:, 1])
        # scale grip factor to friction factor (0...1) for normalisation
        self.friction_factor = (self.grip - 0.09) / (0.82 - 0.09)
 
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert("RGB")
        # get friction factor
        ff = self.friction_factor[index]
        ff = np.float32(ff)
        # transform image 
        img = self.transform(img)
        
        return (img, ff)

    def __len__(self):
        return self.data_len
