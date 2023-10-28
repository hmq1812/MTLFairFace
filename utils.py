import numpy as np
import os
import cv2
import json
import torch
import torchvision
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.annotations = label_path
        self.data_path = data_path
        self.transform = transform

        # Create a label encoder for every categorical attribute
        self.gender_encoder = LabelEncoder()
        self.race_encoder = LabelEncoder()

        # Fit the encoder on the categories and transform the labels to numerical data
        self.annotations['age'] = self.race_encoder.fit_transform(self.annotations['age'].values)
        self.annotations['gender'] = self.gender_encoder.fit_transform(self.annotations['gender'].values)
        self.annotations['race'] = self.race_encoder.fit_transform(self.annotations['race'].values)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_path, self.annotations.iloc[idx, 0]) 
        image = Image.open(img_name)
        
        # Here you can add whatever information you need from the csv to the labels dictionary.
        labels = {'age': torch.tensor(self.annotations.iloc[idx, 1], dtype=torch.long), 
                    'gender': torch.tensor(self.annotations.iloc[idx, 2], dtype=torch.long), 
                    'race': torch.tensor(self.annotations.iloc[idx, 3], dtype=torch.long)}

            

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
        

class FairFaceLoader(BaseDataLoader):
    def __init__(self, data_path, label_path, batch_size=128, train=True, shuffle=True, drop_last=False, transform=None):
        """
        Args:
            label_path (string): Path to the CSV file with annotations.
            data_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(FairFaceLoader, self).__init__(batch_size, train, shuffle, drop_last)
        
        self.data_path = data_path
        self.label = pd.read_csv(label_path)

        self.transform = transform if transform else torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create a dataset-like structure
        self.dataset = CustomDataset(data_path=self.data_path, label_path=self.label, transform=self.transform)

        # You can further split your data into training and validation here if needed

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last)

        self._len = len(self.label)


    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self.dataloader)


if __name__ == "__main__":
    F = FairFaceLoader("FairFaceData/fairface-img-margin025-trainval/", "FairFaceData/fairface_label_val.csv")
    print(F.dataset[1])
    for inputs, labels in F:  # labels should be a list of labels for each task.
        print(inputs)
        print(labels)
        break