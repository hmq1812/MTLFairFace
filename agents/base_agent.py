import torch


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