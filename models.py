import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_V2_S_Weights


class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_list, dropout_rate=0.5):
        super(MultiTaskModel, self).__init__()
        # # Load a pre-trained ResNet-34 model
        # shared_backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # modules = list(shared_backbone.children())[:-1]  # all layers except the last fully connected layer
        # self.shared_backbone = nn.Sequential(*modules)
        # output_feature_size = shared_backbone.fc.in_features  # get the no. of in_features in fc layer

        # Load a pre-trained EfficientNetV2 model
        shared_backbone = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Remove the classifier (fully connected layer)
        self.shared_backbone = nn.Sequential(*list(shared_backbone.children())[:-1])

        # Find out the output features size based on the backbone architecture
        output_feature_size = shared_backbone.classifier[1].in_features

        num_age, num_gender, num_race = num_classes_list
        # Task-specific layers, Each task will have its own classification head
        self.age_layer = nn.Sequential(
            nn.Linear(output_feature_size, 128),
            nn.BatchNorm1d(128),  # Batch Normalization layer
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Dropout layer
            nn.Linear(128, num_age)
        )

        self.gender_layer = nn.Sequential(
            nn.Linear(output_feature_size, 128),
            nn.BatchNorm1d(128),  # Batch Normalization layer
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Dropout layer
            nn.Linear(128, num_gender)
        )
        
        self.race_layer = nn.Sequential(
            nn.Linear(output_feature_size, 128),
            nn.BatchNorm1d(128),  # Batch Normalization layer
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Dropout layer
            nn.Linear(128, num_race)  
        )

    def forward(self, x):
        # Forward pass through the shared backbone
        shared_features = self.shared_backbone(x)
        shared_features = torch.flatten(shared_features, 1)  # Flatten the features from the backbone

        # Forward pass through task-specific layers
        out_age = self.age_layer(shared_features)
        out_gender = self.gender_layer(shared_features)
        out_race = self.race_layer(shared_features)

        # Return dictionary output
        return {
            'age': out_age,
            'gender': out_gender,
            'race': out_race
        }


if __name__ == "__main__":
    num_classes_list = [9, 2, 7]  

    # Create the model
    multi_task_model = MultiTaskModel(num_classes_list=num_classes_list)

    # Move model to appropriate device (e.g., GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    multi_task_model = multi_task_model.to(device)

