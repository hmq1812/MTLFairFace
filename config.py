# config.py
# Model config
AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDER_CLASSES = ["Male", "Female"]
RACE_CLASSES = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]

MODEL_CONFIG = {
    'task_name': ['age', 'gender', 'race'],
    'no_classes_per_task': [9, 2, 7],
    'dropout_rate': 0.5
}


# Paths
TRAIN_DATA_PATH = "Data/FairFaceData/fairface-img-margin025-trainval/"
# TRAIN_LABEL_FILE = "Data/FairFaceData/fairface_label_train_encoded.csv"
TRAIN_LABEL_FILE = "Data/FairFaceData/test.csv"

# TRAIN_DATA_PATH = "Data/UTKface_Aligned_cropped/UTKFace/"
# TRAIN_LABEL_FILE = "Data/UTKface_Aligned_cropped/utk_label_train_encoded.csv"

VAL_DATA_PATH = "Data/FairFaceData/fairface-img-margin025-trainval/"
# VAL_LABEL_FILE = "Data/FairFaceData/fairface_label_val_encoded.csv"
VAL_LABEL_FILE = "Data/FairFaceData/test.csv"

TEST_DATA_PATH = "Data/FairFaceData/fairface-img-margin025-trainval/"
TEST_LABEL_FILE = "Data/FairFaceData/fairface_label_test_encoded.csv"

SAVE_PATH = "model"
MODEL_PATH = "model/EfficientNet-ImageNet-131123/last_model.pth"

# Model Parameters
LOSS_WEIGHT = None
ASSIGN_LABEL_THRESHOLD = 0.8
ENTROPY_WEIGHT = 0.5

# Train Parameters
OPTIMIZER = {
    'optimizer_type': 'SGD',
    'lr': 0.01,
    'momentum': 0.9
}

NUM_EPOCHS = 10

# Data parameters
BATCH_SIZE = 8
SHUFFLE = True
DROP_LAST = True
TRANSFORM = None


# Flags
SAVE_MODEL = True
SAVE_HISTORY = True
