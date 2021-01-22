import os

CLASSES = ["CI", "CWI", "MCWI", "SOI"]
N_FOLDS = 5
SEED = 42
BASE_PATH = os.getenv("BASE_PATH", ".")
DEVICE = "cpu"
BATCH_SIZE = 4