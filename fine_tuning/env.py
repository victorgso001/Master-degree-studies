import os

ORIG_INPUT_DATASET = 'Food-11'
BASE_PATH = 'dataset'

TRAIN_BASE = 'training'
TEST_BASE = 'evaluation'
VALIDATION_BASE = 'validation'

CLASSES = ['Bread', 'Dairy Product', 'Dessert', 'Egg', 'Fried Food', 'Meat',
           'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']

BATCH_SIZE = 16

LENCODER_PATH = os.path.sep.join(['output', 'lencoder.cpickle'])

BASE_CSV_PATH = 'output'

MODEL_PATH = os.path.sep.join(['output', 'output.cpickle'])

UNFROZEN_PLOT_PATH = os.path.sep.join(['output', 'unfrozen.png'])
WARMUP_PLOT_PATH = os.path.sep.join(['output', 'warmup.png'])