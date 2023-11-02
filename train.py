import argparse
from agents import FairFaceMultiTaskAgent
from dataset import FairFaceLoader
from loss import MultiTaskLoss, MissingLabelLoss
import config

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--verbose', action='store_true', default=True)

    return parser.parse_args()

def train(args):
    train_data = FairFaceLoader(config.TRAIN_DATA_PATH, config.TRAIN_LABEL_FILE, batch_size=config.BATCH_SIZE)
    test_data = FairFaceLoader(config.TEST_DATA_PATH, config.TRAIN_LABEL_FILE, batch_size=config.BATCH_SIZE)
    loss_fn = MissingLabelLoss(task_names=config.CLASS_NAME, loss_weights=config.LOSS_WEIGHT, threshold=config.ASSIGN_LABEL_THRESHOLD)
    agent = FairFaceMultiTaskAgent(loss_fn, config.CLASS_NAME, config.CLASS_LIST, config.LOSS_WEIGHT)

    agent.train(
        train_data=train_data,
        test_data=test_data,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        save_history=config.SAVE_HISTORY,
        save_path=config.SAVE_PATH,
        verbose=args.verbose
    )


def eval(args):
    data = FairFaceLoader(config.TEST_DATA_PATH, config.TEST_LABEL_FILE, batch_size=config.BATCH_SIZE)

    agent = FairFaceMultiTaskAgent(config.CLASS_LIST)
    agent.load_model(config.MODEL_PATH)
    
    metrics = agent.eval(data)

    print('Overall Accuracy:', metrics['accuracy'])
    for task, acc in metrics['task_accuracies'].items():
        print(f"{task.capitalize()} Accuracy: {acc:.4f}")

def main():
    args = parse_args()
    if args.train:
        train(args)
    elif args.eval:
        eval(args)
    else:
        print('No flag is assigned. Please assign either \'--train\' or \'--eval\'.')

if __name__ == '__main__':
    main()
