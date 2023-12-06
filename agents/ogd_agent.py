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
from .base_agent import EarlyStopping
from .continual_agent import ContinualLearningAgent
from .mtl_agent import MultiTaskAgent

from ogd.memory import *
from ogd.utils import *


class OGDAgent(ContinualLearningAgent):
    """
    Agent for Orthogonal Gradient Descent (OGD) in a continual learning context.
    """

    def __init__(self, optimizer_config, model_config, loss_weights=None, memory_size=100):
        super().__init__(optimizer_config, model_config, loss_weights)
        self.memory = OGDMemory(memory_size=memory_size)
        self.ogd_basis = None
        # Additional initialization as required

    def train(self, train_data, val_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        self.model.train()
        # early_stopping = EarlyStopping(patience=10, min_delta=0.01)  
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

            # if early_stopping(val_eval_metrics['accuracy']):
            #     print("Early stopping triggered")
            #     break

            # Update OGD basis at the end of each epoch
            self.update_ogd_basis(train_data)

        self.save_model(last_model_path)

    def update_ogd_basis(self, train_data):
        """
        Updates the OGD basis using the provided training data.
        Args:
            train_data: Training data loader.
        """
        # Switch model to training mode
        self.model.train()

        for data, targets, _ in train_data:
            # Ensure data is on the correct device
            data = data.to(self.device)

            # Move each set of labels to the correct device
            targets_on_device = {task: labels.to(self.device) for task, labels in targets.items()}

            # Forward pass
            outputs = self.model(data)

            # Compute total loss for all tasks
            total_loss = 0.0
            for task, output in outputs.items():
                task_loss, _ = self.loss_fn.compute_loss(output, targets_on_device[task], task)
                total_loss += task_loss

            # Zero gradients before backward pass
            self.optimizer.zero_grad()

            # Backward pass to compute gradients
            total_loss.backward()
            
            # Extract the gradients and convert them to a vector
            gradients = parameters_to_grad_vector(self.model.parameters())
            # Store gradients in memory
            self.memory.store(gradients)

            # Clear gradients after extraction
            self.optimizer.zero_grad()

        # Retrieve all stored gradients
        all_gradients = self.memory.retrieve_all()
        # Stack all gradients to form a matrix
        gradient_matrix = torch.stack(all_gradients).transpose(0, 1)
        # Orthonormalize the gradient matrix
        self.ogd_basis = orthonormalize(gradient_matrix).to(self.device)


    def optimizer_step(self):
        """
        Overrides the optimizer step to include OGD logic.
        """
        # Ensure model parameters and gradients are on the correct device
        self.model.to(self.device)

        # Extract current gradients as a vector
        current_grads = parameters_to_grad_vector(self.model.parameters()).to(self.device)

        # Project the current gradients onto the OGD basis, if it exists
        if self.ogd_basis is not None and self.ogd_basis.nelement() > 0:
            projected_grads = project_vec(current_grads, self.ogd_basis)
            # Replace the original gradients with the projected gradients
            grad_vector_to_parameters(projected_grads, self.model.parameters())

        # Perform a standard optimization step
        self.optimizer.step()

        # Optionally, clear the gradients after the step (if not done automatically)
        self.model.zero_grad()

    def update_model(self, total_loss, optimizer):
        """
        Updates the model parameters based on the total loss.
        Args:
            total_loss (Tensor): The computed loss for a batch of data.
        """
        # Zero the gradients before the backward pass
        self.optimizer.zero_grad()

        # Compute gradients (backpropagation)
        total_loss.backward()

        # Apply gradient clipping if needed
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Perform the optimizer step (including OGD adjustments)
        self.optimizer_step()



if __name__ == "__main__":
    pass