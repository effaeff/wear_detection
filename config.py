"""Config file"""

from pytorchutils.globals import nn

OPT = False

DATA_DIR = '/raid/drills_dataset/01_raw'
PROCESSED_DIR = '/raid/drills_dataset/RT100XF_processed'
RESULTS_DIR = '/cephfs/drills_dataset/results/xf_trans'
MODELS_DIR = '/cephfs/drills_dataset/models'
INFER_DIR = '/raid/drills_dataset/04_inference'
AUG_DIR = '/cephfs/drills_dataset/04_augmentations'
DATA_LABELS = ['features', 'target'] # Assume to have order: features, targets
OUTPUT_SIZE = 3
PROCESSED_DIM = (448, 576)
RESIZE_SIZE = (256, 256)

data_config = {
    'data_dir': DATA_DIR,
    'processed_dir': PROCESSED_DIR,
    'processed_dim': PROCESSED_DIM,
    'resize_size': RESIZE_SIZE,
    'results_dir': RESULTS_DIR,
    'data_labels': DATA_LABELS,
    'output_size': OUTPUT_SIZE,
    'random_seed': 1234,
    'batch_size': 24,
    'test_size': 0.2 # Percentage of dataset
}
model_config = {
    'models_dir': MODELS_DIR,
    'output_size': OUTPUT_SIZE,
    # 'arch': 'deeplabv3_resnet50',
    # 'arch': 'fcn_resnet50',
    # 'arch': 'lraspp_mobilenet_v3_large',
    'arch': 'vgg16',
    'vgg_bn': True,
    'init': 'xavier_uniform_',
    'init_layers': (nn.ConvTranspose2d),
    'optimizer': 'RMSprop',
    'loss': 'FocalLoss',
    # 'loss': 'CrossEntropyLoss',
    # 'loss': 'DiceLoss',
    'max_iter': 701,
    'learning_rate': 0.001001143265388551,
    'optim_betas': [0.0, 0.999],
    # 'reg_lambda': 0.0001,
    'reg_lambda': 0,
    # 'drop_rate': 0.1,
    'drop_rate': 0,
    'pretrained': True,
    # Early stopping stuff
    'patience': 50,
    'max_problem': True
}
