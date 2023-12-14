import os
import torch
import torchvision
import pandas as pd
from PIL import Image
from math import ceil
import random

import config

class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.annotations = pd.read_csv(label_path)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        # Get the file name and labels
        img_file, age, gender, race = self.annotations.loc[:, ['file', 'age', 'gender', 'race']].iloc[idx]
        img_name = os.path.join(self.data_path, img_file)
        image = Image.open(img_name)
        
        # Convert labels to tensors
        age = torch.tensor(int(age), dtype=torch.long)
        gender = torch.tensor(int(gender), dtype=torch.long)
        race = torch.tensor(int(race), dtype=torch.long)
        
        labels = {
            'age': age, 
            'gender': gender, 
            'race': race
        }

        if self.transform:
            image = self.transform(image)

        return image, labels, img_file

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
    def __init__(self, data_path, label_path, batch_size=128, shuffle=True, drop_last=False, transform=None):
        """
        Args:
            label_path (string): Path to the CSV file with annotations.
            data_path (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super(FairFaceLoader, self).__init__(batch_size, shuffle, drop_last)
        
        self.data_path = data_path
        self.label_path = label_path


        self.transform = transform if transform else torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create a dataset-like structure
        self.dataset = CustomDataset(data_path=self.data_path, label_path=self.label_path, transform=self.transform)
        self._len = len(self.dataset)//batch_size if drop_last else ceil(len(self.dataset)//batch_size)
        # You can further split your data into training and validation here if needed

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last)


    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self.dataloader)


class ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, train_data_path, train_label_path, replay_data_path, replay_label_path, replay_ratio=0.5, transform=None):
        self.train_dataset = CustomDataset(train_data_path, train_label_path, transform)
        self.replay_dataset = CustomDataset(replay_data_path, replay_label_path, transform)
        self.replay_ratio = replay_ratio

    def __len__(self):
        return int(len(self.train_dataset) * (1 + self.replay_ratio))

    def __getitem__(self, idx):
        if idx < len(self.train_dataset):
            return self.train_dataset[idx]
        else:
            replay_idx = random.randint(0, len(self.replay_dataset) - 1)
            return self.replay_dataset[replay_idx]

class ReplayDataLoader(BaseDataLoader):
    def __init__(self, train_data_path, train_label_path, replay_data_path, replay_label_path, replay_ratio=0.5, batch_size=128, shuffle=True, drop_last=False, transform=None):
        super(ReplayDataLoader, self).__init__(batch_size, shuffle, drop_last)
        
        self.transform = transform if transform else torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset = ReplayDataset(train_data_path, train_label_path, replay_data_path, replay_label_path, replay_ratio, transform=self.transform)
        self._len = len(self.combined_dataset)//batch_size if drop_last else ceil(len(self.dataset)/batch_size)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                drop_last=drop_last)
    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self.dataloader)


if __name__ == "__main__":
    F = ReplayDataLoader(config.TRAIN_DATA_PATH_ML, config.TRAIN_LABEL_FILE_ML, config.TRAIN_DATA_PATH, config.TRAIN_LABEL_FILE, replay_ratio=config.REPLAY_RATIO, batch_size=config.BATCH_SIZE)
    # F = FairFaceLoader(config.TRAIN_DATA_PATH_ML, config.TRAIN_LABEL_FILE_ML, batch_size=16)
    print(F.dataset[1])
    print(len(F))
    # for inputs, labels, _ in F:  # labels should be a list of labels for each task.
    #     print(inputs)
    #     print(labels)
    #     break

            