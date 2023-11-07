import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
from tqdm import tqdm

from models import MultiTaskFairFaceModel


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
    def __init__(self, loss_fn, task_names, num_classes_per_task, loss_weights=None):
        super().__init__()
        # Initialize the shared model for all tasks.
        self.model = MultiTaskFairFaceModel(num_classes_list=num_classes_per_task).to(self.device)

        # Set loss weights if provided, else equal weighting.
        if loss_weights is None:
            self.loss_weights = [1. / len(num_classes_per_task)] * len(num_classes_per_task)
        else:
            self.loss_weights = loss_weights

        self.task_names = task_names
        self.multi_task_loss = loss_fn


    def train(self, train_data, test_data, num_epochs=50, lr=0.1, save_history=False, save_path='.', verbose=False):
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # Set up storage for metrics
        history = {
            'accuracy': [],
            'total_loss': [],
            'task_loss': {task: [] for task in self.task_names}  # Storing task-specific losses
        }

        best_accuracy = 0.0  # Initialize best accuracy
        best_model_path = os.path.join(save_path, 'best_model.pth')  # Path to save the best model
        last_model_path = os.path.join(save_path, 'last_model.pth')  # Path to save the last model

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            task_losses = {task: 0.0 for task in self.task_names}

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}")
                progress_bar = tqdm(train_data, total=len(train_data), desc='Training')

            for inputs, labels in (progress_bar if verbose else train_data):
                inputs = inputs.to(self.device)
                labels = {task: label.to(self.device) for task, label in labels.items()}

                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                total_loss, task_losses = self.multi_task_loss.compute_loss(outputs, labels)
                
                total_loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
      
                optimizer.step()
                epoch_loss += total_loss.item()

                if verbose:
                    progress_bar.set_postfix(epoch_loss=epoch_loss)


            # Store metrics after each epoch
            history['total_loss'].append(epoch_loss)
            for task, loss in task_losses.items():
                history['task_loss'][task].append(loss)

            # Evaluate after each epoch
            eval_metrics = self.eval(test_data, self.task_names)  # Adjusting eval method for multi-task
            history['accuracy'].append(eval_metrics['accuracy'])
            
            # Check and save best model
            if eval_metrics['accuracy'] > best_accuracy:
                best_accuracy = eval_metrics['accuracy']
                self.save_model(best_model_path)  # Save the current best model
                if verbose:
                    print(f"Best model updated with accuracy: {best_accuracy:.4f} at epoch {epoch+1}")

            if verbose:
                print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
                for task, loss in task_losses.items():
                    print(f"{task.capitalize()} Task Loss: {loss:.4f}")

        # Save the last model after all epochs are complete
        self.save_model(last_model_path)
        
        if save_history:
            self._save_history(history, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Define CSV filename
        filename = os.path.join(save_path, 'training_history.csv')

        # Define the CSV column headers
        headers = ['epoch', 'total_loss', 'accuracy'] + [f"{task}_task_loss" for task in history['task_loss'].keys()]

        # Write to CSV
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

            for epoch in range(len(history['accuracy'])):
                row = [epoch+1, history['total_loss'][epoch], history['accuracy'][epoch]]
                for task, losses in history['task_loss'].items():
                    row.append(losses[epoch])
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