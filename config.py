"""Config file"""

from pytorchutils.globals import nn

DATA_DIR = '../data/01_raw'
PROCESSED_DIR = '../data/02_processed'
RESULTS_DIR = '../results'
MODELS_DIR = '../models'
INFER_DIR = '../data/03_inference'
DATA_LABELS = ['features', 'target'] # Assume to have order: features, targets
OUTPUT_SIZE = 3

data_config = {
    'data_dir': DATA_DIR,
    'processed_dir': PROCESSED_DIR,
    'processed_dim': (576, 448),
    'results_dir': RESULTS_DIR,
    'data_labels': DATA_LABELS,
    'output_size': OUTPUT_SIZE,
    'random_seed': 4321,
    'batch_size': 4,
    'test_size': 0.2 # Percentage of dataset
}
model_config = {
    'models_dir': MODELS_DIR,
    'output_size': OUTPUT_SIZE,
    'arch': 'vgg16',
    'vgg_bn': True,
    'init': 'xavier_uniform_',
    'init_layers': (nn.ConvTranspose2d),
    'optimizer': 'Adam',
    'loss': 'BCELoss',
    'max_iter': 402,
    'learning_rate': 1e-3,
    'optim_betas': [0.0, 0.999],
    'reg_lambda': 1e-5,
    'pretrained': True
}
