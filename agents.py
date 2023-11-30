import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import csv
from tqdm import tqdm
from math import ceil

from models import MultiTaskModel
from loss import MultiTaskLoss, PseudoLabelingLoss
import config


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_val_acc = 0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.max_val_acc:
            self.max_val_acc = val_acc
            self.counter = 0
        elif self.max_val_acc - val_acc >= self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


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

    def initialize_optimizer(self, optimizer_type, lr, momentum=None):
        pass
    

class MultiTaskAgent(BaseAgent):
    def __init__(self, optimizer_config, model_config, loss_weights=None):
        super().__init__()
        # Unpack config params
        self.optim_type = optimizer_config['optimizer_type']
        self.lr = optimizer_config['lr']
        self.optim_momentum = optimizer_config['momentum']
        self.task_names = model_config['task_name']
        self.num_classes_per_task = model_config['no_classes_per_task']
        self.dropout_rate = model_config['dropout_rate']

        self.model = MultiTaskModel(self.num_classes_per_task, self.dropout_rate).to(self.device)

        if loss_weights is None:
            self.loss_weights = [1.0 / len(self.task_names)] * len(self.task_names)
        else:
            self.loss_weights = loss_weights


        self.loss_fn = MultiTaskLoss(self.task_names, self.loss_weights)
        self.optimizer = self.initialize_optimizer(self.optim_type, self.lr, self.optim_momentum)

    def initialize_optimizer(self, optimizer_type, lr, momentum=None):
        if optimizer_type == 'SGD':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        # Additional optimizers as needed
        else:
            raise ValueError("Unsupported optimizer type")

    def train(self, train_data, val_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        self.model.train()
        early_stopping = EarlyStopping(patience=10, min_delta=0.01)  
        history = self.init_history()

        best_accuracy = 0.0
        best_model_path = os.path.join(save_path, 'best_model.pth')
        last_model_path = os.path.join(save_path, 'last_model.pth')

        for epoch in range(num_epochs):
            epoch_loss, task_losses, task_accuracies, task_sample_counters = self.train_epoch(train_data, self.optimizer, verbose)
            self.update_history(history, epoch_loss, task_losses, task_accuracies, task_sample_counters)
            val_eval_metrics = self.eval(val_data, self.task_names)
            self.update_validation_history(history, val_eval_metrics)

            best_accuracy = self.save_best_model(val_eval_metrics, best_accuracy, best_model_path)
            if verbose:
                self.print_epoch_stats(epoch, num_epochs, epoch_loss, task_losses, task_accuracies, val_eval_metrics, task_sample_counters)

            if save_history:
                self._save_history(history, save_path)

            if early_stopping(val_eval_metrics['accuracy']):
                print("Early stopping triggered")
                break

        self.save_model(last_model_path)

    def init_history(self):
        return {
            'train_accuracy': [],
            'val_accuracy': [],
            'train_task_accuracy': {task: [] for task in self.task_names},
            'val_task_accuracy': {task: [] for task in self.task_names},
            'total_loss': [],
            'task_loss': {task: [] for task in self.task_names},
        }

    def train_epoch(self, train_data, optimizer, verbose):
        epoch_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_names}
        task_accuracies = {task: 0 for task in self.task_names}
        task_sample_counters = {task: 0 for task in self.task_names}

        progress_bar = tqdm(train_data, total=len(train_data), desc='Training') if verbose else train_data
        for inputs, labels, _ in progress_bar:
            inputs, labels = self.prepare_batch(inputs, labels)
            total_loss, batch_task_losses, outputs = self.compute_batch_loss(inputs, labels)
            self.update_model(total_loss, optimizer)
            self.update_batch_stats(outputs, batch_task_losses, labels, task_losses, task_accuracies, task_sample_counters)

            epoch_loss += total_loss.item()
            if verbose:
                progress_bar.set_postfix(epoch_loss=epoch_loss)
        return epoch_loss, task_losses, task_accuracies, task_sample_counters

    def prepare_batch(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = {task: label.to(self.device) for task, label in labels.items()}
        return inputs, labels

    def compute_batch_loss(self, inputs, labels):
        outputs = self.model(inputs)
        total_loss, batch_task_losses = self.loss_fn.compute_loss(outputs, labels)
        return total_loss, batch_task_losses, outputs

    def update_model(self, total_loss, optimizer):
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

    def update_batch_stats(self, outputs, batch_task_losses, labels, task_losses, task_accuracies, task_sample_counters):
        for task in self.task_names:
            task_losses[task] += batch_task_losses[task]  # batch_task_losses[task] is already a float
            _, predicted = torch.max(outputs[task], 1)
            correct = (predicted == labels[task]).sum().item()
            task_accuracies[task] += correct
            task_sample_counters[task] += labels[task].size(0)

    def update_history(self, history, epoch_loss, task_losses, task_accuracies, task_sample_counters):
        total_samples = sum(task_sample_counters.values())
        history['total_loss'].append(epoch_loss / total_samples)
        history['train_accuracy'].append(sum(task_accuracies.values()) / total_samples)
        for task in self.task_names:
            history['task_loss'][task].append(task_losses[task] / task_sample_counters[task])
            history['train_task_accuracy'][task].append(task_accuracies[task] / task_sample_counters[task])

    def update_validation_history(self, history, val_eval_metrics):
        history['val_accuracy'].append(val_eval_metrics['accuracy'])
        for task in self.task_names:
            history['val_task_accuracy'][task].append(val_eval_metrics['task_accuracies'][task])

    def save_best_model(self, val_eval_metrics, best_accuracy, best_model_path):
        current_accuracy = val_eval_metrics['accuracy']
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            self.save_model(best_model_path)
        return best_accuracy

    def print_epoch_stats(self, epoch, num_epochs, epoch_loss, task_losses, task_accuracies, val_eval_metrics, task_sample_counters):
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss / sum(task_sample_counters.values()):.4f}, Train Accuracy: {sum(task_accuracies.values()) / sum(task_sample_counters.values()):.4f}, Val Accuracy: {val_eval_metrics['accuracy']:.4f}")
        for task in self.task_names:
            print(f"{task.capitalize()} - Train Task Loss: {task_losses[task] / task_sample_counters[task]:.4f}, Train Task Accuracy: {task_accuracies[task] / task_sample_counters[task]:.4f}, Val Task Accuracy: {val_eval_metrics['task_accuracies'][task]:.4f}")

    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        filename = os.path.join(save_path, 'training_history.csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = self.get_history_headers()
            writer.writerow(headers)

            # Iterate through each epoch and write its data
            num_epochs = len(history['total_loss'])
            for epoch in range(num_epochs):
                row = [epoch + 1]
                row.append(history['total_loss'][epoch])
                row.append(history['train_accuracy'][epoch])
                row.append(history['val_accuracy'][epoch])
                for task in self.task_names:
                    row.append(history['task_loss'][task][epoch])
                    row.append(history['train_task_accuracy'][task][epoch])
                    row.append(history['val_task_accuracy'][task][epoch])
                writer.writerow(row)

    def get_history_headers(self):
        return ['epoch', 'total_loss', 'train_accuracy', 'val_accuracy'] + \
               [f"{task}_task_loss" for task in self.task_names] + \
               [f"{task}_train_accuracy" for task in self.task_names] + \
               [f"{task}_val_accuracy" for task in self.task_names]

    def format_history_data(self, history):
        num_epochs = len(history['train_accuracy'])
        for epoch in range(num_epochs):
            yield [epoch + 1] + \
                  [history['total_loss'][epoch], history['train_accuracy'][epoch], history['val_accuracy'][epoch]] + \
                  [history['task_loss'][task][epoch] for task in self.task_names] + \
                  [history['train_task_accuracy'][task][epoch] for task in self.task_names] + \
                  [history['val_task_accuracy'][task][epoch] for task in self.task_names]

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
            for inputs, labels, _ in data_loader:
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


class ContinualLearningAgent(MultiTaskAgent):
    def __init__(self, optimizer_config, model_config, loss_weights=None, threshold=0.8, entropy_weight=0.5):
        # Initialize the base MultiTaskAgent with the given configurations
        super().__init__(optimizer_config, model_config, loss_weights)
        self.threshold = threshold

        # Override the loss function with PseudoLabelingLoss specific to continual learning
        self.loss_fn = PseudoLabelingLoss(
            task_names=self.task_names,
            loss_weights=self.loss_weights,
            threshold=threshold,
            entropy_weight=entropy_weight
        )

    def train(self, train_data, val_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        self.model.train()
        early_stopping = EarlyStopping(patience=10, min_delta=0.01)  
        history = self.init_history()

        best_accuracy = 0.0
        best_model_path = os.path.join(save_path, 'best_model.pth')
        last_model_path = os.path.join(save_path, 'last_model.pth')

        for epoch in range(num_epochs):
            epoch_loss, task_losses, task_accuracies, task_sample_counters, pseudo_labels_created = self.train_epoch(train_data, self.optimizer, verbose)
            self.update_history(history, epoch_loss, task_losses, task_accuracies, task_sample_counters)
            val_eval_metrics = self.eval(val_data, self.task_names)
            self.update_validation_history(history, val_eval_metrics)

            # Update label file with pseudo-labels if necessary
            if pseudo_labels_created:
                self.update_label_file(config.TRAIN_LABEL_FILE, pseudo_labels_created, task_name='race')

            best_accuracy = self.save_best_model(val_eval_metrics, best_accuracy, best_model_path)
            if verbose:
                self.print_epoch_stats(epoch, num_epochs, epoch_loss, task_losses, task_accuracies, val_eval_metrics, task_sample_counters)

            if save_history:
                self._save_history(history, save_path)

            if early_stopping(val_eval_metrics['accuracy']):
                print("Early stopping triggered")
                break

            self.save_model(last_model_path)

    def train_epoch(self, train_data, optimizer, verbose):
        self.model.train()
        epoch_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_names}
        task_accuracies = {task: 0 for task in self.task_names}
        task_sample_counters = {task: 0 for task in self.task_names}
        pseudo_labels_created = {}

        progress_bar = tqdm(train_data, total=len(train_data), desc='Training') if verbose else train_data
        for inputs, labels, file_paths in progress_bar:
            inputs, labels = self.prepare_batch(inputs, labels)
            outputs = self.model(inputs)

            total_loss = torch.tensor(0.0, device=inputs.device)

            for task in self.task_names:
                task_output = outputs[task]
                task_label = labels[task]

                total_loss, task_specific_loss, _pseudo_labels = self.compute_task_loss(task_output, task_label, file_paths, task)
                epoch_loss += total_loss.item()
                task_losses[task] += task_specific_loss
                task_sample_counters[task] += labels[task].size(0)

                # Update task accuracies
                _, predicted = torch.max(task_output, 1)
                correct = (predicted == task_label).sum().item()
                task_accuracies[task] += correct

            self.update_model(total_loss, optimizer)
            if verbose:
                progress_bar.set_postfix(epoch_loss=epoch_loss)

        return epoch_loss, task_losses, task_accuracies, task_sample_counters, pseudo_labels_created

    
    def compute_task_loss(self, task_output, task_label, file_paths, task_name):
        # Ensure task_label is a tensor, and convert -1 to False, others to True for the mask
        valid_label_mask = task_label != -1

        # Initialize an empty dictionary for pseudo-labels
        _pseudo_labels = {}

        # Check if there are valid labels
        if valid_label_mask.any():
            masked_task_output = task_output[valid_label_mask]
            masked_task_label = task_label[valid_label_mask]
            total_loss, task_specific_loss = self.loss_fn.compute_loss(masked_task_output, masked_task_label, task_name)
        else:
            total_loss = torch.tensor(0.0, device=task_output.device)
            task_specific_loss = 0.0

        # Compute softmax probabilities and get the maximum probabilities and corresponding labels
        softmax_probs = F.softmax(task_output, dim=1)
        max_probs, pseudo_labels = torch.max(softmax_probs, dim=1)

        # Collect pseudo-labels if applicable
        for file_path, prob, actual_label, pseudo_label in zip(file_paths, max_probs, task_label, pseudo_labels):
            if actual_label == -1 and prob >= self.threshold:
                _pseudo_labels[file_path] = (task_name, pseudo_label.item())

        return total_loss, task_specific_loss, _pseudo_labels

    def update_label_file(label_file_path, pseudo_labels_created, task_name):
        # Read the existing label data
        with open(label_file_path, 'r') as file:
            lines = file.readlines()

        # Build a mapping from filename to line index
        filename_to_line_index = {line.split(',')[0]: i for i, line in enumerate(lines)}

        # Update the lines with the new pseudo-labels
        for filename, new_label in pseudo_labels_created.items():
            if filename in filename_to_line_index:
                line_index = filename_to_line_index[filename]
                parts = lines[line_index].strip().split(',')
                task_index = parts.index(task_name)
                parts[task_index] = str(new_label)
                lines[line_index] = ','.join(parts) + '\n'

        # Write the updated label data back to the file
        with open(label_file_path, 'w') as file:
            file.writelines(lines)

    # The rest of the methods from MultiTaskAgent remain unchanged


if __name__ == "__main__":
    pass