import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from models import MultiTaskFairFaceModel  # This is a hypothetical model you should define based on your tasks.


class BaseAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_data, test_data, num_epochs, save_history, save_path, verbose):
        raise NotImplementedError

    def eval(self, data):
        raise NotImplementedError

    def save_model(self, save_path):
        pass

    def load_model(self, save_path):
        pass


class FairFaceMultiTaskAgent(BaseAgent):
    def __init__(self, num_classes_per_task, loss_weights=None):
        super().__init__()
        # Initialize the shared model for all tasks.
        self.model = MultiTaskFairFaceModel(num_classes_list=num_classes_per_task).to(self.device)

        # Set loss weights if provided, else equal weighting.
        if loss_weights is None:
            self.loss_weights = [1. / len(num_classes_per_task)] * len(num_classes_per_task)
        else:
            self.loss_weights = loss_weights

    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        self.model.train()

        # Criterion for each task
        # criterions = [nn.CrossEntropyLoss() for _ in range(len(self.loss_weights))]
        criterions = [nn.CrossEntropyLoss(), nn.BCELoss(), nn.CrossEntropyLoss()]
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        for epoch in range(num_epochs):
            for inputs, labels in train_data:  # labels should be a list of labels for each task.
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]  # Move each set of labels to the device.
                optimizer.zero_grad()
                accuracy = []

                total_loss = 0
                outputs = self.model(inputs)  # Get outputs for all tasks.
                for i, output in enumerate(outputs):
                    # Calculate and accumulate loss for each task.
                    loss = criterions[i](output, labels[i])
                    total_loss += self.loss_weights[i] * loss

                total_loss.backward()
                optimizer.step()
            
            accuracy.append(self.eval(test_data)['accuracy'])

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))
                # NEED TO ADD MORE LOGGING METRICS: TASK LOSS, TOTAL LOSS

        if save_history:
            self._save_history(accuracy, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for i, h in enumerate(zip(*history)):
            filename = os.path.join(save_path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def save_model(self, save_path='model.pth'):
        """Save the current model parameters to the specified path."""
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path='model.pth'):
        """Load model parameters from the specified path."""
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
        else:
            raise ValueError(f"No such file or directory: '{load_path}'")

    def eval(self, data_loader):
        """Evaluate the model's performance on the provided data loader."""
        self.model.eval()  # Set the model to evaluation mode

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Calculate the number of correct predictions
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == targets).sum().item()

                # Update running totals
                total_loss += loss.item()
                total_correct += correct
                total_samples += targets.size(0)

        average_loss = total_loss / len(data_loader)
        accuracy = total_correct / total_samples

        # Optionally, return more detailed results or statistics
        return {'loss': average_loss, 'accuracy': accuracy}

if __name__ == "__main__":
    pass