from torch.utils.data import Dataset
import data.util_3D as Util
import os
import numpy as np
import scipy.io as sio

class ACDCDataset(Dataset):
    def __init__(self, dataroot, split='train'):
        self.split = split
        self.imageNum = []
        self.dataroot = dataroot

        datapath = os.path.join(dataroot, split, 'data_ED_ES')
        dataFiles = sorted(os.listdir(datapath))
        for isub, dataName in enumerate(dataFiles):
            self.imageNum.append(os.path.join(datapath, dataName))

        self.data_len = len(self.imageNum)
        self.fineSize = [128, 128, 32]

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        dataPath = self.imageNum[index]
        data_ = sio.loadmat(dataPath)
        dataA = data_['image_ED']
        dataB = data_['image_ES']
        label_dataA = data_['label_ED']
        label_dataB = data_['label_ES']

        if self.split == 'test':
            dataName = dataPath.split('/')[-1]
            data_ = sio.loadmat(os.path.join(self.dataroot, self.split, 'data_ED2ES', dataName))
            dataW = data_['image']
            nsample = dataW.shape[-1]
        else:
            nsample = 0

        dataA -= dataA.min()
        dataA /= dataA.std()
        dataA -= dataA.min()
        dataA /= dataA.max()

        dataB -= dataB.min()
        dataB /= dataB.std()
        dataB -= dataB.min()
        dataB /= dataB.max()

        nh, nw, nd = dataA.shape
        sh = int((nh - self.fineSize[0]) / 2)
        sw = int((nw - self.fineSize[1]) / 2)
        dataA = dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        dataB = dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        label_dataA = label_dataA[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]
        label_dataB = label_dataB[sh:sh + self.fineSize[0], sw:sw + self.fineSize[1]]

        if nd >= 32:
            sd = int((nd - self.fineSize[2]) / 2)
            dataA = dataA[..., sd:sd + self.fineSize[2]]
            dataB = dataB[..., sd:sd + self.fineSize[2]]
            label_dataA = label_dataA[..., sd:sd + self.fineSize[2]]
            label_dataB = label_dataB[..., sd:sd + self.fineSize[2]]
        else:
            sd = int((self.fineSize[2] - nd) / 2)
            dataA_ = np.zeros(self.fineSize)
            dataB_ = np.zeros(self.fineSize)
            dataA_[:, :, sd:sd + nd] = dataA
            dataB_[:, :, sd:sd + nd] = dataB
            label_dataA_ = np.zeros(self.fineSize)
            label_dataB_ = np.zeros(self.fineSize)
            label_dataA_[:, :, sd:sd + nd] = label_dataA
            label_dataB_[:, :, sd:sd + nd] = label_dataB
            dataA, dataB = dataA_, dataB_
            label_dataA, label_dataB = label_dataA_, label_dataB_

        [data, label] = Util.transform_augment([dataA, dataB], split=self.split, min_max=(-1, 1))

        return {'M': data, 'F': label, 'MS': label_dataA, 'FS': label_dataB, 'nS':nsample, 'P':dataPath, 'Index': index}
