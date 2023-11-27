import argparse
from agents import MultiTaskAgent, ContinualLearningAgent
from dataset import FairFaceLoader
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Train the FairFace Multi-Task Model")
    parser.add_argument("-v", '--verbose', action='store_true', default=True, help='Print verbose training logs.')
    parser.add_argument("-m", '--missing_label', action='store_true', default=False, help='Use fully labelled data for training.')
    return parser.parse_args()


def train_standard(args, train_data, val_data):
    print('Using standard MultiTaskLoss')
    agent = MultiTaskAgent(config.OPTIMIZER, config.MODEL_CONFIG, config.LOSS_WEIGHT)

    if args.resume:
        print("Loading weights from:", config.MODEL_PATH)
        agent.load_model(config.MODEL_PATH)

    agent.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=config.NUM_EPOCHS,
        save_history=config.SAVE_HISTORY,
        save_path=config.SAVE_PATH,
        verbose=args.verbose
    )


def train_continual(args, train_data, val_data):
    print('Using Loss Function for missing label data')
    agent = ContinualLearningAgent(
        optimizer_config=config.OPTIMIZER, 
        model_config=config.MODEL_CONFIG, 
        loss_weights=config.LOSS_WEIGHT, 
        threshold=config.ASSIGN_LABEL_THRESHOLD, 
        entropy_weight=config.ENTROPY_WEIGHT
    )

    print("Loading weights from:", config.MODEL_PATH)
    agent.load_model(config.MODEL_PATH)

    agent.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=config.NUM_EPOCHS,
        save_history=config.SAVE_HISTORY,
        save_path=config.SAVE_PATH,
        verbose=args.verbose
    )


def train(args):
    if args.missing_label:
        train_data = FairFaceLoader(config.TRAIN_DATA_PATH_ML, config.TRAIN_LABEL_FILE_ML, batch_size=config.BATCH_SIZE)
        val_data = FairFaceLoader(config.VAL_DATA_PATH, config.VAL_LABEL_FILE, batch_size=config.BATCH_SIZE)
        train_continual(args, train_data, val_data)
    else:
        train_data = FairFaceLoader(config.TRAIN_DATA_PATH, config.TRAIN_LABEL_FILE, batch_size=config.BATCH_SIZE)
        val_data = FairFaceLoader(config.VAL_DATA_PATH, config.VAL_LABEL_FILE, batch_size=config.BATCH_SIZE)
        train_standard(args, train_data, val_data)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
