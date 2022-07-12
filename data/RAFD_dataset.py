from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io

class RAFDDataset(Dataset):
    def __init__(self, dataroot, split='test'):
        self.split = split
        self.imageNum = []

        self.datapath = os.path.join(dataroot, split)
        dataFiles = sorted(os.listdir(self.datapath))

        self.imageNum.append([dataFiles[0], dataFiles[1]])
        self.data_len = len(self.imageNum)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fileInfo = self.imageNum[index]
        dataX, dataY = fileInfo[0], fileInfo[1]
        dataXPath = os.path.join(self.datapath, dataX)
        dataYPath = os.path.join(self.datapath, dataY)
        data = io.imread(dataXPath, as_gray=True).astype(float)[:, :, np.newaxis]
        label = io.imread(dataYPath, as_gray=True).astype(float)[:, :, np.newaxis]

        dataX_RGB = io.imread(dataXPath).astype(float)
        dataY_RGB = io.imread(dataYPath).astype(float)

        [data, label] = Util.transform_augment([data, label], split=self.split, min_max=(-1, 1))

        return {'M': data, 'F': label, 'MC': dataX_RGB, 'FC': dataY_RGB, 'nS': 7, 'P':fileInfo, 'Index': index}
