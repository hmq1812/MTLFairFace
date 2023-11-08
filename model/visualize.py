import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_training_logs(df, output_path='training_visualization.png'):
    # Create a figure with three subplots, one for each required plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Total Loss
    axs[0].plot(df['epoch'], df['total_loss'], marker='o', linestyle='-')
    axs[0].set_title('Total Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Total Loss')

    # Accuracy
    axs[1].plot(df['epoch'], df['accuracy'], marker='o', color='green', linestyle='-')
    axs[1].set_title('Accuracy over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')

    # Task Losses
    axs[2].plot(df['epoch'], df['age_task_loss'], marker='o', linestyle='-', label='Age Task Loss')
    axs[2].plot(df['epoch'], df['gender_task_loss'], marker='o', linestyle='-', label='Gender Task Loss')
    axs[2].plot(df['epoch'], df['race_task_loss'], marker='o', linestyle='-', label='Race Task Loss')
    axs[2].set_title('Task-specific Losses over Epochs')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Task Loss')
    axs[2].legend()

    # Adjust layout for better fit
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main(csv_path, output_path):
    try:
        # Read the training log CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        # Plot and save the figures in a single image
        plot_training_logs(df, output_path)
    except FileNotFoundError:
        print(f"The file {csv_path} was not found. Please check the path and try again.")
    except pd.errors.EmptyDataError:
        print(f"The file {csv_path} is empty. Please provide a valid CSV file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize training log.')
    parser.add_argument('-i', '--csv_path', type=str, default='training_history.csv', help='Path to the training log CSV file.')
    parser.add_argument('-o', '--output_path', type=str, default='training_visualization.png', help='Output path for the combined visualization image.')

    args = parser.parse_args()
    main(args.csv_path, args.output_path)
