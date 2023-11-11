import os
import torch
import torchvision
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from math import ceil

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


if __name__ == "__main__":
        F = FairFaceLoader("Data/UTKface_Aligned_cropped/UTKFace", "Data/UTKface_Aligned_cropped/utk_label_train_encoded.csv", batch_size=16, shuffle=False, drop_last=False, transform=None)
        print(F.dataset[1])
        for inputs, labels in F:  # labels should be a list of labels for each task.
            print(inputs)
            print(labels)
            break

            