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
from .mtl_agent import MultiTaskAgent
from .base_agent import EarlyStopping
import config

from ogd.memory import *
from ogd.utils import *


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

                total_loss, task_specific_loss, _pseudo_labels = self.compute_batch_loss(task_output, task_label, file_paths, task)
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

    
    def compute_batch_loss(self, task_output, task_label, file_paths, task_name):
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