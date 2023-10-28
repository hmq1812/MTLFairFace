import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from models import MultiTaskFairFaceModel
from tqdm import tqdm



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
        
        task_names = ['age', 'gender', 'race'] 

        # Criterion for each task (loss function)
        criterions = {
            'age': nn.CrossEntropyLoss(),
            'gender': nn.CrossEntropyLoss(),
            'race': nn.CrossEntropyLoss()
        }

        optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        # Set up storage for metrics
        history = {
            'accuracy': [],
            'total_loss': [],
            'task_loss': {task: [] for task in task_names}  # Storing task-specific losses
        }

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            task_losses = {task: 0.0 for task in task_names}

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}")
                progress_bar = tqdm(train_data, total=len(train_data), desc='Training')

            for inputs, labels in (progress_bar if verbose else train_data):
                inputs = inputs.to(self.device)
                labels = {task: label.to(self.device) for task, label in labels.items()}

                optimizer.zero_grad()
                
                total_loss = 0
                outputs = self.model(inputs)

                for i, task in enumerate(task_names):
                    output_for_task = outputs[task]
                    labels_for_task = labels[task]

                    loss = criterions[task](output_for_task, labels_for_task)
                    task_losses[task] += loss.item()  # Store task-specific loss
                    total_loss += self.loss_weights[i] * loss 

                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

                if verbose:
                    progress_bar.set_postfix(epoch_loss=epoch_loss)

            # Store metrics after each epoch
            history['total_loss'].append(epoch_loss)
            for task, loss in task_losses.items():
                history['task_loss'][task].append(loss)

            # Evaluate after each epoch
            eval_metrics = self.eval(test_data, task_names, criterions)  # Adjusting eval method for multi-task
            history['accuracy'].append(eval_metrics['accuracy'])

            if verbose:
                print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
                for task, loss in task_losses.items():
                    print(f"{task.capitalize()} Task Loss: {loss:.4f}")

        if save_history:
            self._save_history(history, save_path)


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

    def eval(self, data_loader, task_names, criterions):
        """Adjusted evaluation method for multi-task learning."""
        self.model.eval()

        total_correct = {task: 0 for task in task_names}
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = {task: label.to(self.device) for task, label in labels.items()}

                outputs = self.model(inputs)

                for task in task_names:
                    _, predicted = torch.max(outputs[task], 1)
                    correct = (predicted == labels[task]).sum().item()

                    total_correct[task] += correct

                total_samples += labels[task_names[0]].size(0) 

        # Calculate accuracy for each task and the overall accuracy
        accuracies = {task: correct / total_samples for task, correct in total_correct.items()}
        overall_accuracy = sum(accuracies.values()) / len(accuracies)

        return {'accuracy': overall_accuracy, 'task_accuracies': accuracies}

if __name__ == "__main__":
    pass