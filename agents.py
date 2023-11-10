import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
from tqdm import tqdm
from math import ceil

from models import MultiTaskFairFaceModel
import config


class BaseAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_data, val_data, num_epochs, save_history, save_path, verbose):
        raise NotImplementedError

    def eval(self, data, task_names):
        raise NotImplementedError

    def save_model(self, save_path):
        pass

    def load_model(self, save_path):
        pass

class BaseAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_data, val_data, num_epochs, save_history, save_path, verbose):
        raise NotImplementedError

    def eval(self, data_loader, task_names):
        raise NotImplementedError

    def save_model(self, save_path):
        pass

    def load_model(self, save_path):
        pass


class FairFaceMultiTaskAgent(BaseAgent):
    def __init__(self, loss_fn, task_names, num_classes_per_task, loss_weights=None):
        super().__init__()
        self.model = MultiTaskFairFaceModel(num_classes_list=num_classes_per_task, dropout_rate=config.DROPOUT_RATE).to(self.device)

        if loss_weights is None:
            self.loss_weights = {task: 1. / len(task_names) for task in task_names}
        else:
            self.loss_weights = loss_weights

        self.task_names = task_names
        self.multi_task_loss = loss_fn

    def train(self, train_data, val_data, num_epochs=50, lr=0.1, save_history=False, save_path='.', verbose=False):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # Storage for metrics
        history = {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_task_accuracy': {task: [] for task in self.task_names},
            'val_task_accuracy': {task: [] for task in self.task_names},
            'total_loss': [],
            'task_loss': {task: [] for task in self.task_names},
        }

        best_accuracy = 0.0
        best_model_path = os.path.join(save_path, 'best_model.pth')
        last_model_path = os.path.join(save_path, 'last_model.pth')

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            task_losses = {task: 0.0 for task in self.task_names}
            task_accuracies = {task: 0 for task in self.task_names}
            task_sample_counters = {task: 0 for task in self.task_names}

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}")
                progress_bar = tqdm(train_data, total=ceil(len(train_data)/config.BATCH_SIZE), desc='Training')

            for inputs, labels in (progress_bar if verbose else train_data):
                inputs = inputs.to(self.device)
                labels = {task: label.to(self.device) for task, label in labels.items()}

                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                total_loss, task_losses = self.multi_task_loss.compute_loss(outputs, labels)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += total_loss.item()

                # Calculate task-specific accuracies
                for task in self.task_names:
                    _, predicted = torch.max(outputs[task], 1)
                    correct = (predicted == labels[task]).sum().item()
                    task_accuracies[task] += correct
                    task_sample_counters[task] += labels[task].size(0)

                if verbose:
                    progress_bar.set_postfix(epoch_loss=epoch_loss)

            # Store metrics after each epoch
            for task in self.task_names:
                history['train_task_accuracy'][task].append(task_accuracies[task] / task_sample_counters[task])

            overall_train_accuracy = sum(task_accuracies.values()) / sum(task_sample_counters.values())
            history['train_accuracy'].append(overall_train_accuracy)

            # Evaluate on validation set
            val_eval_metrics = self.eval(val_data, self.task_names)
            history['val_accuracy'].append(val_eval_metrics['accuracy'])
            for task in self.task_names:
                history['val_task_accuracy'][task].append(val_eval_metrics['task_accuracies'][task])

            # Check and save the best model
            if val_eval_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_eval_metrics['accuracy']
                self.save_model(best_model_path)
                if verbose:
                    print(f"Best model updated with accuracy: {best_accuracy:.4f} at epoch {epoch+1}")

            if verbose:
                print(f"Epoch {epoch+1} - Train Loss: {epoch_loss / sum(task_sample_counters.values()):.4f}, Train Accuracy: {overall_train_accuracy:.4f}, Val Accuracy: {val_eval_metrics['accuracy']:.4f}")

        # Save the last model after all epochs are complete
        self.save_model(last_model_path)
        
        if save_history:
            self._save_history(history, save_path)

    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        filename = os.path.join(save_path, 'training_history.csv')
        headers = ['epoch', 'total_loss', 'train_accuracy', 'val_accuracy'] \
                  + [f"{task}_task_loss" for task in self.task_names] \
                  + [f"{task}_train_accuracy" for task in self.task_names] \
                  + [f"{task}_val_accuracy" for task in self.task_names]

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

            num_epochs = len(history['train_accuracy'])
            for epoch in range(num_epochs):
                row = [epoch+1,
                       history['total_loss'][epoch],
                       history['train_accuracy'][epoch],
                       history['val_accuracy'][epoch]] \
                      + [history['task_loss'][task][epoch] for task in self.task_names] \
                      + [history['train_task_accuracy'][task][epoch] for task in self.task_names] \
                      + [history['val_task_accuracy'][task][epoch] for task in self.task_names]
                writer.writerow(row)


    def save_model(self, save_path='model.pth'):
        """Save the current model parameters to the specified path."""
        
        # Check if the directory of the save_path exists, if not create it
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path='model.pth'):
        """Load model parameters from the specified path."""
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))
        else:
            raise ValueError(f"No such file or directory: '{load_path}'")

    def eval(self, data_loader, task_names):
        """Adjusted evaluation method for multi-task learning."""
        self.model.eval()

        total_correct = {task: 0 for task in task_names}
        task_sample_counts = {task: 0 for task in task_names}

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = {task: label.to(self.device) for task, label in labels.items()}

                outputs = self.model(inputs)

                for task in task_names:
                    if len(labels[task]) > 0:  # Check if there are labels for the task
                        _, predicted = torch.max(outputs[task], 1)
                        correct = (predicted == labels[task]).sum().item()
                        total_correct[task] += correct
                        task_sample_counts[task] += labels[task].size(0)  # Count samples for each task separately

        # Calculate accuracy for each task
        accuracies = {task: (total_correct[task] / task_sample_counts[task]) if task_sample_counts[task] > 0 else 0 for task in task_names}

        # Calculate the overall accuracy
        if sum(task_sample_counts.values()) > 0:
            overall_accuracy = sum(total_correct.values()) / sum(task_sample_counts.values())
        else:
            overall_accuracy = 0  # Prevent division by zero

        return {'accuracy': overall_accuracy, 'task_accuracies': accuracies}


if __name__ == "__main__":
    pass