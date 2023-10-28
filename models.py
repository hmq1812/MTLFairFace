import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights

# MTL FAIRFACE MODEL 

class MultiTaskFairFaceModel(nn.Module):
    def __init__(self, num_classes_list):
        super(MultiTaskFairFaceModel, self).__init__()
        # Load a pre-trained ResNet-34 model
        shared_backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        # Remove the fully connected layer
        modules = list(shared_backbone.children())[:-1]  # all layers except the last fully connected layer
        self.shared_backbone = nn.Sequential(*modules)

        # Find out the the fully connected layer's input features based on the backbone architecture
        output_feature_size = shared_backbone.fc.in_features  # get the no. of in_features in fc layer

        num_gender, num_age, num_race = num_classes_list
        # Task-specific layers, Each task will have its own classification head
        self.gender_layer = nn.Sequential(
            nn.Linear(output_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_gender)
        )
        
        self.age_layer = nn.Sequential(
            nn.Linear(output_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_age)
        )
        
        self.race_layer = nn.Sequential(
            nn.Linear(output_feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_race)  
        )


    def forward(self, x):
        # Forward pass through the shared backbone
        shared_features = self.shared_backbone(x)
        shared_features = torch.flatten(shared_features, 1)  # Flatten the features from the backbone

        # Forward pass through task-specific layers
        out_gender = self.gender_layer(shared_features)
        out_age = self.age_layer(shared_features)
        out_race = self.race_layer(shared_features)

        # Return dictionary output
        return {
            'gender': out_gender,
            'age': out_age,
            'race': out_race
        }


if __name__ == "__main__":
    # Define the number of classes for each task (e.g., age, gender, race)
    # race_scores_fair (model confidence score) [White, Black, Latino_Hispanic, East, Southeast Asian, Indian, Middle Eastern]
    # race_scores_fair_4 (model confidence score) [White, Black, Asian, Indian]
    # gender_scores_fair (model confidence score) [Male, Female]
    # age_scores_fair (model confidence score) [0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]
    num_classes_list = [9, 2, 7]  

    # Create the model
    multi_task_model = MultiTaskFairFaceModel(num_classes_list=num_classes_list)

    # Move model to appropriate device (e.g., GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    multi_task_model = multi_task_model.to(device)

