import numpy as np
import argparse
from agents import FairFaceMultiTaskAgent
from utils import FairFaceLoader


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--eval', action='store_true')

    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_history', action='store_true')

    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def train(args):
    # NEED TO CHANGE
    train_data = FairFaceLoader("FairFaceData/fairface_label_val.csv", "FairFaceData/fairface-img-margin025-trainval/")
    test_data = FairFaceLoader("FairFaceData/fairface_label_val.csv", "FairFaceData/fairface-img-margin025-trainval/")

    num_classes_single = train_data.num_classes_single
    num_classes_multi = train_data.num_classes_multi
    num_tasks = len(num_classes_multi)
    num_channels = train_data.num_channels

    agent = FairFaceMultiTaskAgent([9,2,7])

    agent.train(train_data=train_data,
                test_data=test_data,
                num_epochs=num_epochs,
                save_history=args.save_history,
                save_path=args.save_path,
                verbose=args.verbose
                )

    if args.save_model:
        agent.save_model(args.save_path)


def eval(args):
    # NEED TO CHANGE
    data = FairFaceLoader("FairFaceData/fairface_label_val.csv", "FairFaceData/fairface-img-margin025-trainval/")

    num_classes_single = data.num_classes_single
    num_classes_multi = data.num_classes_multi
    num_tasks = len(num_classes_multi)
    num_channels = data.num_channels

    agent = FairFaceMultiTaskAgent([9,2,7])

    agent.load_model(args.save_path)
    accuracy = agent.eval(data)

    print('Accuracy: {}'.format(accuracy))


def main():
    args = parse_args()
    if args.train:
        train(args)
    elif args.eval:
        eval(args)
    else:
        print('No flag is assigned. Please assign either \'--train\' or \'--eval\'.')


if __name__ == '__main__':
    # main()
    A = FairFaceMultiTaskAgent([9,2,7])
    train_data = FairFaceLoader("FairFaceData/fairface_label_val.csv", "FairFaceData/fairface-img-margin025-trainval/")
    test_data = FairFaceLoader("FairFaceData/fairface_label_val.csv", "FairFaceData/fairface-img-margin025-trainval/")
    A.train(train_data, test_data)