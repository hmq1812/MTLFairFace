import argparse
from agents import FairFaceMultiTaskAgent
from dataset import FairFaceLoader
import config
from loss import MultiTaskLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the FairFace Multi-Task Model")
    return parser.parse_args()

def eval(args):
    data = FairFaceLoader(config.TEST_DATA_PATH, config.TEST_LABEL_FILE, batch_size=config.BATCH_SIZE)
    # This loss function is just to fill all neeeded params, not actually in use in inference and eval.
    loss_fn = MultiTaskLoss(task_names=config.CLASS_NAME, loss_weights=config.LOSS_WEIGHT)
    agent = FairFaceMultiTaskAgent(loss_fn, config.CLASS_NAME, config.CLASS_LIST, config.LOSS_WEIGHT)
    agent.load_model(config.MODEL_PATH)
    
    metrics = agent.eval(data, config.CLASS_NAME)

    print('Overall Accuracy:', metrics['accuracy'])
    for task, acc in metrics['task_accuracies'].items():
        print(f"{task.capitalize()} Accuracy: {acc:.4f}")

def main():
    args = parse_args()
    eval(args)

if __name__ == '__main__':
    main()
