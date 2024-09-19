from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as img_file:
            magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
            assert magic_num == 2051
            X = np.frombuffer(img_file.read(img_num * row * col), dtype=np.uint8).astype(np.float32).reshape((img_num, row * col))
            X = X.reshape(img_num, row, col, 1)
            X /= 255.0
    
        with gzip.open(label_filename, 'rb') as label_file:
            magic_num, label_num = struct.unpack(">2i", label_file.read(8))
            assert magic_num == 2049
            y = np.frombuffer(label_file.read(label_num), dtype=np.uint8)
    
        self.img = X
        self.label = y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.img[index]), self.label[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.img)
        ### END YOUR SOLUTION