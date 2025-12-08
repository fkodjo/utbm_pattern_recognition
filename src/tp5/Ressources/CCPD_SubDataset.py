import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import re

class CCPD_SubDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.split = split
        self.data_path = data_path
        self.data = os.listdir(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]

        img = cv2.imread(self.data_path+'/'+img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        match = re.search(r"\d+-\d+_\d+-(?P<x1>\d+)&(?P<y1>\d+)_(?P<x2>\d+)&(?P<y2>\d+)-\d+&\d+_\d+&\d+_\d+&\d+_\d+&\d+-.*", img_path)
        x1 = float(match.group('x1'))/img.shape[1]
        y1 = float(match.group('y1'))/img.shape[0]
        x2 = float(match.group('x2'))/img.shape[1]
        y2 = float(match.group('y2'))/img.shape[0]

        img = cv2.resize(img, (256, 256))

        cx = x1 + ((x2 - x1) / 2.0)
        cy = y1 + ((y2 - y1) / 2.0)
        w = x2 - x1
        h = y2 - y1

        bbox = torch.tensor([cx, cy, w, h])

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return transform(img), bbox
