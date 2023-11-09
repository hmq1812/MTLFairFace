import argparse
from agents import FairFaceMultiTaskAgent
from dataset import FairFaceLoader
from loss import MultiTaskLoss, PseudoLabelingLoss
import config

def parse_args():
    parser = argparse.ArgumentParser(description="Train the FairFace Multi-Task Model")
    parser.add_argument("-v", '--verbose', action='store_true', default=True, help='Print verbose training logs.')
    parser.add_argument("-m", '--missing_label', action='store_true', default=False, help='Use fully labelled data for training.')
    parser.add_argument("-r", '--resume', action='store_true', default=False, help='Resume training from weight file.')
    return parser.parse_args()

def train(args):
    train_data = FairFaceLoader(config.TRAIN_DATA_PATH, config.TRAIN_LABEL_FILE, batch_size=config.BATCH_SIZE)
    test_data = FairFaceLoader(config.TEST_DATA_PATH, config.TEST_LABEL_FILE, batch_size=config.BATCH_SIZE)
    
    if args.missing_label:
        print('Using Loss Function for missing label data')
        loss_fn = PseudoLabelingLoss(task_names=config.CLASS_NAME, loss_weights=config.LOSS_WEIGHT, threshold=config.ASSIGN_LABEL_THRESHOLD, entropy_weight=config.ENTROPY_WEIGHT)
    else:
        loss_fn = MultiTaskLoss(task_names=config.CLASS_NAME, loss_weights=config.LOSS_WEIGHT)
        
    
    agent = FairFaceMultiTaskAgent(loss_fn, config.CLASS_NAME, config.CLASS_LIST, config.LOSS_WEIGHT)

    if args.resume:
        print("Loading weight ...")
        agent.load_model(config.MODEL_PATH)

    agent.train(
        train_data=train_data,
        test_data=test_data,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        save_history=config.SAVE_HISTORY,
        save_path=config.SAVE_PATH,
        verbose=args.verbose
    )

def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()
