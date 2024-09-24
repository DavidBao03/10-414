import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        if train:
            img = np.zeros((0, 3, 32, 32))
            label = np.zeros((0,))
            for i in range(1, 6):
                with open(os.path.join(base_folder, f"data_batch_{i}"), "rb") as fo:
                    dict = pickle.load(fo, encoding="bytes")
                    img = np.concatenate((img, dict[b'data'].reshape((-1, 3, 32, 32))), axis=0)
                    label = np.concatenate((label, dict[b'labels']), axis=0)
        else:
            with open(os.path.join(base_folder, "test_batch"), "rb") as fo:
                    dict = pickle.load(fo, encoding="bytes")
                    img = dict[b'data'].reshape((-1, 3, 32, 32))
                    label = np.array(dict[b'labels'], dtype=np.int32)
        self.img = img.astype(np.float32) / 255.0
        self.label = label
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.img[index]), self.label[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.label)
        ### END YOUR SOLUTION
