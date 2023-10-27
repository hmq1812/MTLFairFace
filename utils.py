import numpy as np
import os
import cv2
import json
import torch
import torchvision
import pandas as pd
from PIL import Image

class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.annotations = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])  # adjust the index if necessary
        image = Image.open(img_name)
        
        # Here you can add whatever information you need from the csv to the labels dictionary.
        labels = {'age': self.annotations.iloc[idx, 1],  # adjust the index if necessary
                  'gender': self.annotations.iloc[idx, 2],  # adjust the index if necessary
                  'race': self.annotations.iloc[idx, 3]}  # adjust the index if necessary

        if self.transform:
            image = self.transform(image)

        return image, labels

class BaseDataLoader:
    def __init__(self, batch_size=1, train=True, shuffle=True, drop_last=False):
        pass

    def get_loader(self, loader, prob):
        raise NotImplementedError

    def get_labels(self, task):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def num_classes_single(self):
        raise NotImplementedError

    @property
    def num_classes_multi(self):
        raise NotImplementedError


class MultiTaskDataLoader:
    def __init__(self, dataloaders, prob='uniform'):
        self.dataloaders = dataloaders
        self.iters = [iter(loader) for loader in self.dataloaders]

        if prob == 'uniform':
            self.prob = np.ones(len(self.dataloaders)) / len(self.dataloaders)
        else:
            self.prob = prob

        self.size = sum([len(d) for d in self.dataloaders])
        self.step = 0


    def __iter__(self):
        return self


    def __next__(self):
        if self.step >= self.size:
            self.step = 0
            raise StopIteration

        task = np.random.choice(list(range(len(self.dataloaders))), p=self.prob)

        try:
            data, labels = self.iters[task].__next__()
        except StopIteration:
            self.iters[task] = iter(self.dataloaders[task])
            data, labels = self.iters[task].__next__()

        self.step += 1

        return data, labels, task
        


# Add new DataLoader for FairFace Data
class FairFaceLoader(BaseDataLoader):
    def __init__(self, csv_file, root_dir, batch_size=128, train=True, shuffle=True, drop_last=False, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(FairFaceLoader, self).__init__(batch_size, train, shuffle, drop_last)

        self.fairface_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform else torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create a dataset-like structure
        self.dataset = CustomDataset(data_frame=self.fairface_frame, root_dir=self.root_dir, transform=self.transform)

        # You can further split your data into training and validation here if needed

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last)

        self._len = len(self.fairface_frame)


    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self.dataloader)


if __name__ == "__main__":
    F = FairFaceLoader("FairFaceData/fairface_label_val.csv", "FairFaceData/fairface-img-margin025-trainval/")
    for x in F:
        print(x)
        break