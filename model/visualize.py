import pandas as pd
import matplotlib.pyplot as plt
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def plot_total_loss(data, ax):
    ax.plot(data['epoch'].to_numpy(), data['total_loss'].to_numpy(), label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()

def plot_train_val_accuracy(data, ax):
    ax.plot(data['epoch'].to_numpy(), data['train_accuracy'].to_numpy(), label='Train Accuracy')
    ax.plot(data['epoch'].to_numpy(), data['val_accuracy'].to_numpy(), label='Validation Accuracy')
    ax.set_ylim([0, 1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Validation Accuracy')
    ax.legend()

def plot_task_losses(data, ax):
    ax.plot(data['epoch'].to_numpy(), data['age_task_loss'].to_numpy(), label='Age Task Loss', color='red')
    ax.plot(data['epoch'].to_numpy(), data['gender_task_loss'].to_numpy(), label='Gender Task Loss', color='green')
    ax.plot(data['epoch'].to_numpy(), data['race_task_loss'].to_numpy(), label='Race Task Loss', color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Task Loss')
    ax.set_title('Task Losses')
    ax.legend()


def plot_accuracies(data, ax):
    tasks = ['age', 'gender', 'race']  # Replace with your actual task names
    colors = {'age': 'red', 'gender': 'green', 'race': 'purple'}

    for task in tasks:
        ax.plot(data['epoch'].to_numpy(), data[f'{task}_train_accuracy'].to_numpy(), color=colors[task], label=f'Training Accuracy {task}')
        ax.plot(data['epoch'].to_numpy(), data[f'{task}_val_accuracy'].to_numpy(), color=colors[task], linestyle='dashed', label=f'Validation Accuracy {task}')
    
    ax.set_ylim([0, 1])
    ax.set_title('Training and Validation Accuracy for Each Task')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()

def create_plots(data):
    plt.figure(figsize=(15, 10))

    ax1 = plt.subplot(2, 2, 1)
    plot_total_loss(data, ax1)

    ax2 = plt.subplot(2, 2, 2)
    plot_task_losses(data, ax2)

    ax3 = plt.subplot(2, 2, 3)
    plot_train_val_accuracy(data, ax3)

    ax4 = plt.subplot(2, 2, 4)
    plot_accuracies(data, ax4)

    plt.tight_layout()
    plt.savefig('training_metrics.png')


def main(csv_path):
    try:
        # Load data from CSV
        data = pd.read_csv(csv_path)
        create_plots(data)
    except FileNotFoundError:
        print(f"The file {csv_path} was not found. Please check the path and try again.")
    except pd.errors.EmptyDataError:
        print(f"The file {csv_path} is empty. Please provide a valid CSV file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training log.')
    parser.add_argument('-i', '--csv_path', type=str, default='training_history.csv', help='Path to the training log CSV file.')

    args = parser.parse_args()
    main(args.csv_path)
