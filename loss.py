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


class MissingLabelLoss:
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
            
            # Check if we have a label for this task
            if label is not None:
                softmax_output = F.softmax(output, dim=1)
                max_prob, max_idx = torch.max(softmax_output, dim=1)
                
                # If the max probability is above the threshold, use normal loss
                mask = max_prob > self.threshold
                valid_samples = mask.sum().item()
                
                if valid_samples > 0:
                    loss = self.loss_functions[task](output[mask], label[mask])
                else:
                    loss = torch.tensor(0.0).to(output.device)
                
                # For the rest, use entropy as a penalty
                # Softmax: omputes the softmax of the model's output
                # Log Softmax: improve numerical stability.
                # Entropy Calculation: calculates the negative log likelihood for each class
                # High entropy means that the model is uncertain in its predictions, as the probabilities are spread out across multiple classes
                entropy_loss = -torch.mean(torch.sum(F.softmax(output[~mask], dim=1) 
                                                    * F.log_softmax(output[~mask], dim=1), dim=1))
                
                
                # Combine the losses
                loss = loss + entropy_loss
                
            # If no label, just use entropy loss as a penalty
            else:
                loss = -torch.mean(torch.sum(F.softmax(output, dim=1) 
                                             * F.log_softmax(output, dim=1), dim=1))
            
            task_losses[task] = loss.item()
            total_loss += self.loss_weights[task] * loss
            
        return total_loss, task_losses
