import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss:
    def __init__(self, task_names, loss_weights=None, **kwargs):
        self.task_names = task_names
        self.loss_functions = {task: nn.CrossEntropyLoss(reduction='mean') for task in task_names}
        
        if loss_weights is None:
            self.loss_weights = {task: 1. / len(task_names) for task in task_names}
        else:
            self.loss_weights = {task: weight for task, weight in zip(task_names, loss_weights)}
    
    def compute_loss(self, outputs, labels):
        total_loss = 0
        task_losses = {task: 0.0 for task in self.task_names}

        for task in self.task_names:
            loss = self.loss_functions[task](outputs[task], labels[task])
            task_losses[task] = loss.item()
            total_loss += self.loss_weights[task] * loss
            
        return total_loss, task_losses


class PseudoLabelingLoss:
    def __init__(self, task_names, loss_weights=None, threshold=0.8):
        self.task_names = task_names
        self.threshold = threshold
        self.loss_functions = {task: nn.CrossEntropyLoss(reduction='mean') for task in task_names}
        
        if loss_weights is None:
            self.loss_weights = {task: 1. / len(task_names) for task in task_names}
        else:
            self.loss_weights = {task: weight for task, weight in zip(task_names, loss_weights)}
    
    def compute_loss(self, outputs, labels):
        total_loss = 0
        task_losses = {task: 0.0 for task in self.task_names}

        for task in self.task_names:
            output = outputs[task]
            label = labels[task]
            
            # Check if the label is not a special value indicating missing label
            valid_label_mask = label != -1
            
            if valid_label_mask.any():
                softmax_output = F.softmax(output[valid_label_mask], dim=1)
                max_prob, _ = torch.max(softmax_output, dim=1)
                pseudo_label_mask = max_prob > self.threshold
                
                valid_samples = pseudo_label_mask.sum().item()
                
                if valid_samples > 0:
                    # Only compute the loss for valid samples with pseudo labels
                    loss = self.loss_functions[task](output[valid_label_mask][pseudo_label_mask], label[valid_label_mask][pseudo_label_mask])
                else:
                    loss = torch.tensor(0.0).to(output.device)
                
                # Calculate entropy loss for samples without pseudo labels
                entropy_loss = -torch.mean(torch.sum(F.softmax(output[valid_label_mask][~pseudo_label_mask], dim=1) 
                                                    * F.log_softmax(output[valid_label_mask][~pseudo_label_mask], dim=1), dim=1))
                # Combine the losses
                loss += entropy_loss
            else:
                # If no valid label, just use entropy loss as a penalty
                # Softmax: omputes the softmax of the model's output
                # Log Softmax: improve numerical stability.
                # Entropy Calculation: calculates the negative log likelihood for each class
                # High entropy means that the model is uncertain in its predictions, as the probabilities are spread out across multiple classes
                loss = -torch.mean(torch.sum(F.softmax(output, dim=1) 
                                            * F.log_softmax(output, dim=1), dim=1))
            
            task_losses[task] = loss.item()
            total_loss += self.loss_weights[task] * loss
            
        return total_loss, task_losses

                
