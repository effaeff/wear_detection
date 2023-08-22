"""Main script for running the segmentation learning task"""

import shutil
import misc

import numpy as np

from pytorchutils.fcn8s import FCNModel as FCNModel
from pytorchutils.ahg import AHGModel
from pytorchutils.hg import HGModel
from pytorchutils.segmentation import Segmentation
from weardetection.dataprocessor import DataProcessor
from weardetection.trainer import Trainer
from pytorchutils.globals import nn

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from config import (
    OPT,
    DATA_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    INFER_DIR,
    DATA_LABELS,
    PROCESSED_DIR,
    data_config,
    model_config,
    INFER_DIR
)

def objective(config):
    """Wrapper for training runs"""
    if OPT:
        lr = config['lr']
        batch_size = int(config['batch_size'])

        data_config['batch_size'] = batch_size
        model_config['learning_rate'] = lr
        model_config['reg_lambda'] = config["reg_lambda"]
        model_config['drop_rate'] = config["drop_rate"]

    data_processor = DataProcessor(data_config)
    # model = Segmentation(model_config)
    # model = FCNModel(model_config)
    model = nn.DataParallel(AHGModel(model_config))
    # model = AHGModel(model_config)
    # model = HGModel(model_config)
    trainer = Trainer(model_config, model, data_processor)
    trainer.get_batches_fn = data_processor.get_batches
    if OPT:
        acc = trainer.train(validate_every=1, save_every=0, save_eval=False, verbose=False)
    else:
        acc = trainer.train(validate_every=5, save_every=1, save_eval=True, verbose=True)
    # trainer.infer(INFER_DIR)

    print(f"Config: {config}, acc: {acc:.2f}")

    return {'loss': 100 - acc, 'params': config, 'status': STATUS_OK}

def main():
    """Main method"""

    # shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
    misc.gen_dirs(
        [DATA_DIR, RESULTS_DIR, MODELS_DIR] +
        [f'{INFER_DIR}/test/{data_lbl}/1' for data_lbl in DATA_LABELS] +
        [f'{PROCESSED_DIR}/train/{data_lbl}/1' for data_lbl in DATA_LABELS] +
        [f'{PROCESSED_DIR}/test/{data_lbl}/1' for data_lbl in DATA_LABELS]
    )

    if not OPT:
        config = {
            'batch_size': 32,
            'lr': 0.001001143265388551,
            # 'lr': 0.00051143265388551,
            'drop_rate': 0,
            'reg_lambda': 0
        }
        objective(config)
        quit()

    search_space = {
        "lr": hp.uniform("lr", 0.000001, 0.001),
        "drop_rate": hp.uniform("drop_rate", 0.01, 0.2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 0.001),
        "batch_size": hp.randint("batch_size", 2, 32)
    }

    trials = Trials()
    best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=75, trials=trials)
    np.save(f"{RESULTS_DIR}/hyperopt_best.npy", best)
    print(f"Finished hyperparameter tuning. Best config:\n{best}")


if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
