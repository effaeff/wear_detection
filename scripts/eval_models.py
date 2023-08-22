"""Script for evaluating already trained models with different checkpoints"""

import re
from glob import glob
import numpy as np
import misc
import pathlib

from pytorchutils.globals import torch, nn
from pytorchutils.fcn8s import FCNModel
from pytorchutils.segmentation import Segmentation
from pytorchutils.hg import HGModel
from pytorchutils.ahg import AHGModel
from weardetection.dataprocessor import DataProcessor
from weardetection.trainer import Trainer

from config import data_config, model_config, MODELS_DIR


def main():
    """Main method"""

    data_processor = DataProcessor(data_config)

    total_dirs = glob(f'/cephfs/drills_dataset/models/xf_trans/')
    # total_dirs.sort(key=lambda f: int(re.search(r'\d+', f.split('_')[-1]).group()))

    best = 0

    for directory in total_dirs:
        print(f"Processing model directory: {directory}")
        # n_measurements = int(re.search(r'\d+', directory.split('_')[-1]).group())
        # checkpoints = glob(f'{directory}*AHGModel*')
        # checkpoints = glob(f'{directory}*HGModel*')
        checkpoints = glob(f'{directory}*DataParallel*')
        # checkpoints = glob(f'{directory}*Segmentation*')
        # checkpoints = glob(f'{directory}*FCNModel*')
        checkpoints.sort(key=lambda f: int(re.search(r'\d+', f.split('_')[-2]).group()))

        accuracies = np.zeros(len(checkpoints))
        stds = np.zeros(len(checkpoints))
        for idx, checkpoint in enumerate(checkpoints):
            epoch = int(re.search(r'\d+', checkpoint.split('_')[-2]).group())
            model = nn.DataParallel(AHGModel(model_config))
            # model = Segmentation(model_config)
            # model = FCNModel(model_config)
            # model = HGModel(model_config)
            # print(f"checkpoint file: {checkpoint}")
            state = torch.load(checkpoint)
            model.load_state_dict(state['state_dict'], strict=False)
            trainer = Trainer(model_config, model, data_processor)
            accuracy, std = trainer.validate(epoch, save_eval=False, verbose=False)
            accuracies[idx] = accuracy
            stds[idx] = std
            print(f'epoch: {epoch}, accuracy: {accuracy:.2f} +- {std:.2f}')
        np.save(f'{directory}accuracy_progression.npy', np.array([accuracies, stds]))
        best = np.argmax(accuracies)

        pathlib.Path(f'{directory}/best').mkdir(parents=True, exist_ok=True)
        checkpoint = f'{directory}/DataParallel_0_epoch{best}_checkpoint.pth.tar'
        # checkpoint = f'{directory}/Segmentation_0_epoch{best}_checkpoint.pth.tar'
        # checkpoint = f'{directory}/FCNModel_0_epoch{best}_checkpoint.pth.tar'
        # checkpoint = f'{directory}/HGModel_0_epoch{best}_checkpoint.pth.tar'
        model = nn.DataParallel(AHGModel(model_config))
        # model = FCNModel(model_config)
        # model = Segmentation(model_config)
        # model = HGModel(model_config)
        print(f"checkpoint file: {checkpoint}")
        state = torch.load(checkpoint)
        model.load_state_dict(state['state_dict'], strict=False)
        trainer = Trainer(model_config, model, data_processor)
        __, __ = trainer.validate(best, save_eval=True, verbose=True)

if __name__ == '__main__':
    misc.to_local_dir(__file__)
    main()
