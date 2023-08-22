"""Script to compose a dataset including augments using a specific augmentation factor"""

import os
import re
import shutil
import numpy as np
import matplotlib.pyplot as plt

import argparse

from tqdm import tqdm
from glob import glob

from config import PROCESSED_DIR, AUG_DIR

def main():
    """Main method"""

    parser = argparse.ArgumentParser()
    parser.add_argument("data_percentage", type=float)
    args = parser.parse_args()
    original_data_percentage = args.data_percentage

    print(f"Composing train dataset using {original_data_percentage*100}% original data")

    old_aug_f = glob(f'{PROCESSED_DIR}/train/features/1/*_aug.npy')
    old_aug_t = glob(f'{PROCESSED_DIR}/train/target/1/*_aug.npy')
    old_diff_f = glob(f'{PROCESSED_DIR}/train/features/1/*_diff.npy')
    old_diff_t = glob(f'{PROCESSED_DIR}/train/target/1/*_diff.npy')

    for f_list in [old_aug_f, old_aug_t, old_diff_f, old_diff_t]:
        for fname in f_list:
            os.remove(fname)

    n_train = len(os.listdir(f'{PROCESSED_DIR}/train/features/1'))
    augments = int((n_train // original_data_percentage - n_train))# // 2)
    augments = int(2 * round(augments / 2))

    print(f'Number of augments: {augments}')

    aug_names = sorted(
        os.listdir(f'{AUG_DIR}/features_aug'), key=lambda x: int(re.search('\d+', x).group())
    )
    diff_names = sorted(
        os.listdir(f'{AUG_DIR}/features_diff'), key=lambda x: int(re.search('\d+', x).group())
    )

    # print("Copy augmentations based on image manupulations...")
    # for fname in tqdm(aug_names[:augments]):
        # shutil.copyfile(f'{AUG_DIR}/features_aug/{fname}', f'{PROCESSED_DIR}/train/features/1/{fname}')
        # shutil.copyfile(
            # f'{AUG_DIR}/target_aug/{os.path.splitext(fname)[0]}.npy',
            # f'{PROCESSED_DIR}/train/target/1/{os.path.splitext(fname)[0]}.npy'
        # )


    # Balance samples
    abrasive_cnt = 0
    adhesive_cnt = 0

    for fname in tqdm(diff_names[::-1]):
        trgt = np.load(f'{AUG_DIR}/target_diff/{os.path.splitext(fname)[0]}.npy')
        adhesive_involved = len(trgt[np.where(trgt==1)].flatten()) > 20
        if adhesive_involved:
            if adhesive_cnt < augments//2: # adhesive wear involved and still necessary
                shutil.copyfile(
                    f'{AUG_DIR}/features_diff/{fname}',
                    f'{PROCESSED_DIR}/train/features/1/{fname}'
                )
                shutil.copyfile(
                    f'{AUG_DIR}/target_diff/{os.path.splitext(fname)[0]}.npy',
                    f'{PROCESSED_DIR}/train/target/1/{os.path.splitext(fname)[0]}.npy'
                )
                adhesive_cnt += 1
        else:
            if abrasive_cnt < augments//2:
                shutil.copyfile(
                    f'{AUG_DIR}/features_diff/{fname}',
                    f'{PROCESSED_DIR}/train/features/1/{fname}')
                shutil.copyfile(
                    f'{AUG_DIR}/target_diff/{os.path.splitext(fname)[0]}.npy',
                    f'{PROCESSED_DIR}/train/target/1/{os.path.splitext(fname)[0]}.npy'
                )
                abrasive_cnt += 1
        if abrasive_cnt + adhesive_cnt >= augments:
            break

    # print("Copy diffusion samples...")
    # for fname in tqdm(diff_names[:augments]):
        # shutil.copyfile(f'{AUG_DIR}/features_diff/{fname}', f'{PROCESSED_DIR}/train/features/1/{fname}')
        # shutil.copyfile(
            # f'{AUG_DIR}/target_diff/{os.path.splitext(fname)[0]}.npy',
            # f'{PROCESSED_DIR}/train/target/1/{os.path.splitext(fname)[0]}.npy'
        # )

if __name__ == "__main__":
    main()
