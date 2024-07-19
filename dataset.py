# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:38:56 2023

@author: spisak
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision


class twoPerspectiveDataset(Dataset):
    def __init__(self,csv_file,transform=None):
        self.dataFrame = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        le = len(self.dataFrame)
        return le
    def __getitem__(self, index):
        leftEye = self.dataFrame.iloc[index].leftEye
        opposite = self.dataFrame.iloc[index].Opposite
        
        leftEye = Image.open(leftEye)
        opposite = Image.open(opposite)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(),torchvision.transforms.Resize((64,64))])
        if self.transform:
            leftEye = self.transform(leftEye)/255
            opposite = self.transform(opposite)/255
        return leftEye,opposite
    
def makeDataset(csv,verbose=False,seed=2147533647,arms=False):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.random_split(twoPerspectiveDataset(csv_file= csv),(0.8,0.2),gen)