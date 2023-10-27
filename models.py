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
        num_ftrs = shared_backbone.fc.in_features  # get the no. of in_features in fc layer

        # Each task will have its own classification head
        # Create them based on the number of tasks and their specific number of classes.
        self.task_heads = nn.ModuleList([
            nn.Linear(num_ftrs, num_classes) for num_classes in num_classes_list
        ])

    def forward(self, x):
        # Forward pass through the shared backbone
        shared_features = self.shared_backbone(x)
        shared_features = torch.flatten(shared_features, 1)  # Flatten the features from the backbone

        # Forward pass through each task-specific head
        logits = [task_head(shared_features) for task_head in self.task_heads]

        return logits  # It returns a list of outputs for each task

        # # Forward pass through the base model
        # x = self.base_model(x)

        # # Task-specific forward passes
        # out_gender = self.fc_gender(x)
        # out_race = self.fc_race(x)
        # out_age = self.fc_age(x)

        # # The model will output a dictionary of outputs
        # return {
        #     'gender': out_gender,
        #     'race': out_race,
        #     'age': out_age
        # }



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

