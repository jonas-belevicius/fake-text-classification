import os
import torch
import spacy
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
utils_dir = os.path.join(project_dir, "utils")
data_dir = os.path.join(project_dir, "data")
sys.path.append(utils_dir)

RANDOM_SEED = 1
DATA_DIR = os.path.join(project_dir, "data")
TRAIN_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
TRAIN_LR = 1e-4
FINE_TUNE_LR = 1e-5
BATCH_SIZE = 8
NUM_WORKERS = 4
CHECKPOINT_PATH = os.path.join(project_dir, "model_checkpoints")
LOGS_PATH = os.path.join(project_dir, "training_logs")
LOGS_NAME = "bert"
DROPOUT = 0.3
GRADIENT_CLIP = 1
DEVICE = torch.device("cuda")
MAX_SEQ_LEN = 512
WEIGHT_DECAY = 1e-4

NLP = spacy.load("en_core_web_sm")

stop_words = spacy.lang.en.stop_words.STOP_WORDS
