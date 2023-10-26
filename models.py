import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class _Encoder(nn.Module):
    def __init__(self, layers):
        super(_Encoder, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)

        return x


class _Decoder(nn.Module):
    def __init__(self, output_size):
        super(_Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128*8*8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.layers(x)

        return x


class _Model(nn.Module):
    def __init__(self, output_size, encoder):
        super(_Model, self).__init__()
        self.encoder = encoder
        self.decoder = _Decoder(output_size=output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def Model(num_classes, num_channels):
    layers = [
        nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    ]

    if isinstance(num_classes, list):
        encoders = [_Encoder(layers=layers) for _ in num_classes]
        return [_Model(output_size=cls, encoder=encoder) for cls, encoder in zip(num_classes, encoders)]
    else:
        encoder = _Encoder(layers=layers)
        return _Model(output_size=num_classes, encoder=encoder)


class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_list):
        super(MultiTaskModel, self).__init__()
        # Load a pre-trained ResNet-34 model
        self.resnet = models.resnet34(pretrained=True)
        
        # Remove the last fully-connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Define new layers for multi-task learning
        # Num_classes_list is a list of integers, each representing the number of classes for a specific task. For example: [7, 2, 5] for race, gender, and age groups respectively.
        self.fc_layers = nn.ModuleList([nn.Linear(self.resnet[-1][-1].in_features, num_classes) for num_classes in num_classes_list])
        
    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # Flatten the features
        
        # Number of outputs = number of tasks
        outputs = [fc_layer(x) for fc_layer in self.fc_layers]
        
        return outputs  # List of outputs for different tasks


if __name__ == "__main__":
    # Example usage:
    # Say we're handling three tasks (race, gender, age), and we have different numbers of classes for each
    num_classes_list = [7, 2, 5]  # just an example, actual numbers based on the dataset
    model = MultiTaskModel(num_classes_list=num_classes_list)
