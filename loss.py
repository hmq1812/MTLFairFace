import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, task_names, loss_weights=None, **kwargs):
        self.task_names = task_names
        self.loss_functions = {task: nn.CrossEntropyLoss(reduction='mean') for task in task_names}
        if loss_weights is None:
            # If no specific weights are provided, assign equal weights to each task
            self.loss_weights = {task: 1.0 / len(task_names) for task in task_names}
        else:
            if len(loss_weights) != len(task_names):
                raise ValueError("Length of loss_weights must match the number of tasks.")
            if not all(isinstance(weight, (float, int)) for weight in loss_weights):
                raise TypeError("All loss weights must be numeric values.")
            self.loss_weights = dict(zip(task_names, loss_weights))

    
    def compute_loss(self, outputs, labels):
        total_loss = 0
        task_losses = {task: 0.0 for task in self.task_names}

        for task in self.task_names:
            loss = self.loss_functions[task](outputs[task], labels[task])
            task_losses[task] = loss.item()
            total_loss += self.loss_weights[task] * loss
            
        return total_loss, task_losses


class PseudoLabelingLoss:
    def __init__(self, task_names, loss_weights=None, threshold=0.8, entropy_weight=0.5):
        self.task_names = task_names
        self.threshold = threshold
        self.loss_functions = {task: nn.CrossEntropyLoss(reduction='mean') for task in task_names}
        
        # Initialize loss weights
        if loss_weights is None:
            self.loss_weights = {task: 1. / len(task_names) for task in task_names}
        else:
            self.loss_weights = {task: weight for task, weight in zip(task_names, loss_weights)}
        
        # Weight for the entropy loss to balance it with cross-entropy loss
        self.entropy_weight = entropy_weight
    
    def compute_loss(self, output, label, task):
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=output.device)
        task_loss = torch.tensor(0.0, device=output.device)

        # Check if the label is not a special value indicating missing label
        valid_label_mask = label != -1

        if valid_label_mask.any():
            # Select the subset of outputs and labels that have valid labels
            valid_outputs = output[valid_label_mask]
            valid_labels = label[valid_label_mask]

            # Compute loss for valid labels
            task_loss = self.loss_functions[task](valid_outputs, valid_labels)
            total_loss += task_loss

            # Compute entropy loss for samples that are not pseudo labeled
            softmax_output = F.softmax(valid_outputs, dim=1)
            max_prob, _ = torch.max(softmax_output, dim=1)
            pseudo_label_mask = max_prob > self.threshold
            non_pseudo_label_mask = ~pseudo_label_mask

            if non_pseudo_label_mask.any():
                entropy_loss = -torch.mean(torch.sum(F.log_softmax(valid_outputs[non_pseudo_label_mask], dim=1) * softmax_output[non_pseudo_label_mask], dim=1))
                total_loss += self.entropy_weight * entropy_loss

        else:
            # Use entropy loss as a penalty for all samples if no valid labels are present
            entropy_loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1) * F.softmax(output, dim=1), dim=1))
            total_loss += self.entropy_weight * entropy_loss

        return total_loss, task_loss.item()

                
if __name__ == "__main__":
    task_names = ['age', 'gender', 'race']
    loss_weights = [0.3, 0.3, 0.4]  # Ensure these are numerical values
    loss_fn = MultiTaskLoss(task_names=task_names, loss_weights=None)
    print("Final loss weights:", loss_fn.loss_weights)


