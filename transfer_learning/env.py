import os

ORIG_INPUT_DATASET = 'Food-5k'
BASE_PATH = 'dataset'

TRAIN_BASE = 'training'
TEST_BASE = 'evaluation'
VALIDATION_BASE = 'validation'

CLASSES = ['non_food', 'food']

BATCH_SIZE = 32

LENCODER_PATH = os.path.sep.join(['output', 'lencoder.cpickle'])

BASE_CSV_PATH = 'output'

MODEL_PATH = os.path.sep.join(['output', 'output.cpickle'])