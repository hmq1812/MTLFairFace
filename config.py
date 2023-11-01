# config.py

# Paths
TRAIN_DATA_PATH = "FairFaceData/fairface-img-margin025-trainval/"
TEST_DATA_PATH = "FairFaceData/fairface-img-margin025-trainval/"
TRAIN_LABEL_FILE = "FairFaceData/fairface_label_train.csv"
TEST_LABEL_FILE = "FairFaceData/fairface_label_val.csv"
SAVE_PATH = "."
MODEL_PATH = "model.pth"

# Train Parameters
NUM_EPOCHS = 20
CLASS_LIST = [2, 9, 7]
LOSS_WEIGHT = None
LEARNING_RATE = 0.01

# Data parameters
BATCH_SIZE = 128
SHUFFLE = True
DROP_LAST = True
TRANSFORM = None


# Flags
SAVE_MODEL = True
SAVE_HISTORY = True
