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


class SingleTaskAgent(BaseAgent):
    def __init__(self, num_classes, num_channels):
        super(SingleTaskAgent, self).__init__()
        self.model = Model(num_classes=num_classes, num_channels=num_channels).to(self.device)


    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        accuracy = []

        for epoch in range(num_epochs):
            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(test_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'history.json')

        with open(filename, 'w') as f:
            json.dump(history, f)


    def eval(self, data):
        correct = 0
        total = 0

        with torch.no_grad():
            self.model.eval()

            for inputs, labels in data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)
                correct += (predict_labels == labels).sum().item()

            self.model.train()

            return correct / total


    def save_model(self, save_path='.'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, 'model')

        torch.save(self.model.state_dict(), filename)


    def load_model(self, save_path='.'):
        if os.path.isdir(save_path):
            filename = os.path.join(save_path, 'model')
            self.model.load_state_dict(torch.load(filename))


class StandardAgent(SingleTaskAgent):
    def __init__(self, num_classes_single, num_classes_multi, multi_task_type, num_channels):
        if multi_task_type == 'binary':
            super(StandardAgent, self).__init__(num_classes=num_classes_single, num_channels=num_channels)
            self.eval = self._eval_binary
            self.num_classes = num_classes_single
        elif multi_task_type == 'multiclass':
            super(StandardAgent, self).__init__(num_classes=num_classes_single, num_channels=num_channels)
            self.eval = self._eval_multiclass
            self.num_classes = num_classes_multi
        else:
            raise ValueError('Unknown multi-task type: {}'.format(multi_task_type))


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for i, h in enumerate(zip(*history)):
            filename = os.path.join(save_path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def _eval_binary(self, data):
        correct = [0 for _ in range(self.num_classes)]
        total = 0

        with torch.no_grad():
            self.model.eval()

            for inputs, labels in data.get_loader():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predict_labels = torch.max(outputs.detach(), 1)

                total += labels.size(0)

                for c in range(self.num_classes):
                    correct[c] += ((predict_labels == c) == (labels == c)).sum().item()

            self.model.train()

            return [c / total for c in correct]


    def _eval_multiclass(self, data):
        num_tasks = len(self.num_classes)
        correct = [0 for _ in range(num_tasks)]
        total = [0 for _ in range(num_tasks)]

        with torch.no_grad():
            self.model.eval()

            for t in range(num_tasks):
                task_labels = data.get_labels(t)
                for inputs, labels in data.get_loader(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predict_labels = torch.max(outputs[:, task_labels].detach(), 1)

                    total[t] += labels.size(0)
                    correct[t] += (predict_labels == labels).sum().item()

            self.model.train()

            return [c / t for c, t in zip(correct, total)]


class MultiTaskSeparateAgent(BaseAgent):
    def __init__(self, num_classes, num_channels, task_prob=None):
        super(MultiTaskSeparateAgent, self).__init__()
        self.num_tasks = len(num_classes)
        self.task_prob = task_prob
        self.models = [model.to(self.device) for model in Model(num_classes=num_classes, num_channels=num_channels)]


    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        for model in self.models:
            model.train()

        if self.task_prob is None:
            dataloader = train_data.get_loader('multi-task')
        else:
            dataloader = train_data.get_loader('multi-task', prob=self.task_prob)

        criterion = nn.CrossEntropyLoss()
        optimizers = [optim.SGD(model.parameters(), lr=0.1) for model in self.models]
        accuracy = []

        for epoch in range(num_epochs):
            for inputs, labels, task in dataloader:
                model = self.models[task]
                optimizer = optimizers[task]

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(test_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, save_path)


    def _save_history(self, history, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for i, h in enumerate(zip(*history)):
            filename = os.path.join(save_path, 'history_class{}.json'.format(i))

            with open(filename, 'w') as f:
                json.dump(h, f)


    def eval(self, data):
        correct = [0 for _ in range(self.num_tasks)]
        total = [0 for _ in range(self.num_tasks)]

        with torch.no_grad():
            for t, model in enumerate(self.models):
                model.eval()

                for inputs, labels in data.get_loader(t):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predict_labels = torch.max(outputs.detach(), 1)

                    total[t] += labels.size(0)
                    correct[t] += (predict_labels == labels).sum().item()

                model.train()

            return [c / t for c, t in zip(correct, total)]


    def save_model(self, save_path='.'):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for t, model in enumerate(self.models):
            filename = os.path.join(save_path, 'model{}'.format(t))
            torch.save(model.state_dict(), filename)


    def load_model(self, save_path='.'):
        if os.path.isdir(save_path):
            for t, model in enumerate(self.models):
                filename = os.path.join(save_path, 'model{}'.format(t))
                model.load_state_dict(torch.load(filename))


class MultiTaskJointAgent(MultiTaskSeparateAgent):
    """
    MultiTaskJointAgent can only be used in tasks that share the same inputs.
    Currently it can only apply to CIFAR-10 multi-task experiments.
    CIFAR-100 and Omniglot multi-task experiments are not applicable.
    """

    def __init__(self, num_classes, multi_task_type, num_channels, loss_weight=None):
        if multi_task_type == 'multiclass':
            raise ValueError('Multi-task type \'multiclass\' is not suitable to MultiTaskJointAgent.')

        super(MultiTaskJointAgent, self).__init__(num_classes, num_channels)

        if loss_weight is None:
            self.loss_weight = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        else:
            self.loss_weight = torch.Tensor(loss_weight).to(self.device)


    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        for model in self.models:
            model.train()

        dataloader = train_data.get_loader()
        criterion = nn.CrossEntropyLoss()

        parameters = []
        for model in self.models:
            parameters += model.parameters()
        parameters = set(parameters)
        optimizer = optim.SGD(parameters, lr=0.1)

        accuracy = []

        for epoch in range(num_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss = 0.

                for t, model in enumerate(self.models):
                    outputs = model(inputs)
                    loss += self.loss_weight[t] * criterion(outputs, (labels == t).long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy.append(self.eval(test_data))

            if verbose:
                print('[Epoch {}] Accuracy: {}'.format(epoch+1, accuracy[-1]))

        if save_history:
            self._save_history(accuracy, save_path)



# ADD FAIRFACE MLT AGENT

class FairFaceMultiTaskAgent(BaseAgent):
    def __init__(self, num_classes_per_task, num_channels, loss_weights=None):
        super().__init__()
        # Initialize the shared model for all tasks.
        self.model = MultiTaskFairFaceModel(num_classes_per_task=num_classes_per_task, num_channels=num_channels).to(self.device)

        # Set loss weights if provided, else equal weighting.
        if loss_weights is None:
            self.loss_weights = [1. / len(num_classes_per_task)] * len(num_classes_per_task)
        else:
            self.loss_weights = loss_weights

    def train(self, train_data, test_data, num_epochs=50, save_history=False, save_path='.', verbose=False):
        self.model.train()

        # Criterion for each task (assuming all are classification).
        criterions = [nn.CrossEntropyLoss() for _ in range(len(self.loss_weights))]
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        for epoch in range(num_epochs):
            for inputs, labels in train_data:  # labels should be a list of labels for each task.
                inputs = inputs.to(self.device)
                labels = [label.to(self.device) for label in labels]  # Move each set of labels to the device.
                optimizer.zero_grad()

                total_loss = 0
                outputs = self.model(inputs)  # Get outputs for all tasks.
                for i, output in enumerate(outputs):
                    # Calculate and accumulate loss for each task.
                    loss = criterions[i](output, labels[i])
                    total_loss += self.loss_weights[i] * loss

                total_loss.backward()
                optimizer.step()

            if verbose:
                print('[Epoch {}]'.format(epoch+1))
                # NEED TO ADD MORE LOGGING METRICS: TASK LOSS, TOTAL LOSS

        # TODO: After training, you may save the model and any relevant training history.
        if save_history:
            # self._save_history(accuracy, save_path)
            pass



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

