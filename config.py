# config.py

# Paths
TRAIN_DATA_PATH = "FairFaceData/fairface-img-margin025-trainval/"
TEST_DATA_PATH = "FairFaceData/fairface-img-margin025-trainval/"
TRAIN_LABEL_FILE = "FairFaceData/fairface_label_train.csv"
TEST_LABEL_FILE = "FairFaceData/fairface_label_val.csv"
SAVE_PATH = "model"
MODEL_PATH = "model/best_model.pth"

# Model Parameters
CLASS_LIST = [2, 9, 7]
CLASS_NAME = ['age', 'gender', 'race']
LOSS_WEIGHT = None
ASSIGN_LABEL_THRESHOLD = 0.8

# Train Parameters
LEARNING_RATE = 0.01
NUM_EPOCHS = 50

# Data parameters
BATCH_SIZE = 32
SHUFFLE = True
DROP_LAST = True
TRANSFORM = None


# Flags
SAVE_MODEL = True
SAVE_HISTORY = True
