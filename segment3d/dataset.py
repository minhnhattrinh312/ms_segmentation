import glob
from torch.utils.data import Dataset
import torch
import numpy as np


class RandomFlip:
    """
    Xoay nao theo truc Z flip
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, volume, mask):
        if torch.rand(1) < self.p:
            volume = volume.flip(1)
            mask = mask.flip(1)

        if torch.rand(1) < self.p:
            volume = volume.flip(2)
            mask = mask.flip(2)
        return volume, mask


class ISBILoader(Dataset):
    def __init__(
        self,
        train_path="",
        transform=None,
        list_subject=[],
        extraction_step=24,
        patch_size=96,
        extract=False,
    ):
        if list_subject:
            self.listName = list_subject
        else:
            self.listName = glob.glob(train_path)
        self.transform = transform
        self.extraction_step = extraction_step
        self.patch_size = patch_size
        self.extract = extract

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        subject = np.load(self.listName[idx])
        try:
            volume, mask = torch.from_numpy(subject["image"]), torch.from_numpy(subject["mask"])
        except:
            print(self.listName[idx])
        # volume, mask = torch.from_numpy(subject["image"]), torch.from_numpy(subject["mask"])
        ####################### augmentation data ##############################
        if self.transform:
            for transform in self.transform:
                volume, mask = transform(volume, mask)

        if self.extract:
            start_indexes = (
                self.extraction_step * torch.randint(3, (1,)),
                self.extraction_step * torch.randint(5, (1,)),
                self.extraction_step * torch.randint(3, (1,)),
            )
            valid_index = (slice(None),) + tuple(slice(start_index, start_index + self.patch_size) for start_index in start_indexes)
            volume, mask = volume[valid_index], mask[valid_index]
        return volume, mask
