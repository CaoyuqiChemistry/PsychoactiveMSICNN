from torch.utils import data
import os
import numpy as np

class MyDataSetClass(data.Dataset):
    def __init__(self,root, t = None):
        self.label_class = {'阿片类':0, '芬太尼类':1, '精神刺激剂类':2, '镇定剂类':3, '致幻剂类':4}
        self.classes = {0:'阿片类', 1:'芬太尼类', 2:'精神刺激剂类', 3:'镇定剂类', 4:'致幻剂类'}
        self.t = t
        self.imgs = os.listdir(root)
        self.img_path = [os.path.join(root,img) for img in self.imgs]

    def __getitem__(self, index):
        data = np.load(self.img_path[index])
        data = data.astype(np.float32)
        if self.t:
            data = self.t(data)
        label = self.label_class[self.imgs[index].split('.')[0].split('_')[0]]
        return data, label

    def __len__(self):
        return len(self.img_path)
