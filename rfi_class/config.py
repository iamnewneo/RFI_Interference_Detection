import os

CLASSES = ["CI", "CWI", "MCWI", "SOI"]
N_FOLDS = 5
SEED = 42
BASE_PATH = os.getenv("BASE_PATH", ".")
DEVICE = "cuda"
BATCH_SIZE = 256
MAX_EPOCHS = 15