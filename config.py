# config.py
# Define the classes
AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDER_CLASSES = ["Male", "Female"]
RACE_CLASSES = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"] 
CLASS_NAME = ['age', 'gender', 'race']
CLASS_LIST = [9, 2, 7]

# Paths
# TRAIN_DATA_PATH = "Data/FairFaceData/fairface-img-margin025-trainval/"
TRAIN_DATA_PATH = "Data/UTKface_Aligned_cropped/UTKFace/"
TEST_DATA_PATH = "Data/FairFaceData/fairface-img-margin025-trainval/"

# TRAIN_LABEL_FILE = "Data/FairFaceData/fairface_label_train_encoded.csv"
TRAIN_LABEL_FILE = "Data/UTKface_Aligned_cropped/utk_label_train_encoded.csv"
TEST_LABEL_FILE = "Data/FairFaceData/fairface_label_val_encoded.csv"

SAVE_PATH = "model"
MODEL_PATH = "model/best_model_101123.pth"

# Model Parameters
LOSS_WEIGHT = None
ASSIGN_LABEL_THRESHOLD = 0.8
ENTROPY_WEIGHT = 0.5

# Train Parameters
LEARNING_RATE = 0.01
NUM_EPOCHS = 50

# Data parameters
BATCH_SIZE = 128
SHUFFLE = True
DROP_LAST = True
TRANSFORM = None


# Flags
SAVE_MODEL = True
SAVE_HISTORY = True
