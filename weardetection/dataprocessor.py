"""Data processing methods"""

import re
import os
import random
from pathlib import Path

import cv2
from PIL import Image

import numpy as np
import torchvision.datasets as dset
import matplotlib.pyplot as plt

from pytorchutils.globals import nn
from pytorchutils.globals import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.model_selection import train_test_split

from sklearn.metrics import jaccard_score


class WearDataset(torch.utils.data.Dataset):
    """PyTorch Dataset to store grain data"""
    def __init__(self, path_features, path_target=None):
        # self.data_features = dset.ImageFolder(
            # root=path_features,
            # transform=Compose([
                # ToTensor(),
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
        # )
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        f_names = sorted(os.listdir(f'{path_features}/1'), key=lambda name: int(name.split('_')[0]))
        self.data_features = np.array([
            transform(Image.open(f'{path_features}/1/{f_name}'))
            for f_name in f_names
        ])
        t_names = sorted(os.listdir(f'{path_target}/1'), key=lambda name: int(name.split('_')[0]))
        self.data_targets = None
        if path_target is not None:
            self.data_targets = np.array([
                np.load(f'{path_target}/1/{t_name}')
                for t_name in t_names
            ])

    def __getitem__(self, index):
        features = self.data_features[index]

        targets = []
        if self.data_targets is not None:
            targets = self.data_targets[index]
            targets = nn.functional.one_hot(torch.LongTensor(targets), num_classes=3).permute(2, 0, 1).float()

        item = {'F': features, 'T': targets}
        return item

    def __len__(self):
        return len(self.data_features) # Assume that both datasets have equal length


class DataProcessor():
    """Class for data processor"""
    def __init__(self, config):
        self.config = config
        self.random_seed = self.config.get('random_seed', 1234)
        self.processed_dim = self.config['processed_dim']
        self.batch_size = self.config.get('batch_size', 4)
        self.data_dir = self.config['data_dir']
        self.processed_dir = self.config['processed_dir']
        self.results_dir = self.config['results_dir']
        self.data_lbls = self.config['data_labels']

        if not any(
            Path(f'{self.processed_dir}/train/{self.data_lbls[0]}/1').iterdir()
        ):
            self.process_raw()

        train_dataset = WearDataset(
            f'{self.processed_dir}/train/{self.data_lbls[0]}',
            f'{self.processed_dir}/train/{self.data_lbls[1]}'
        )
        test_dataset = WearDataset(
            f'{self.processed_dir}/test/{self.data_lbls[0]}',
            f'{self.processed_dir}/test/{self.data_lbls[1]}'
        )

        self.train_data = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=0)
        self.test_data = DataLoader(test_dataset, self.batch_size, shuffle=False, num_workers=0)

    def read_raw(self, data_dir):
        """Reading raw image files"""
        tool_dirs = os.listdir(data_dir)
        data = []
        for directory in tool_dirs:
            filenames = [
                filename for filename in os.listdir(f'{data_dir}/{directory}')
                if not (filename.endswith('AN.png') or filename.endswith('AN.jpg'))
            ]
            for filename in filenames:
                # print(filename)
                feature = cv2.resize(
                    cv2.imread(f'{data_dir}/{directory}/{filename}'), self.processed_dim
                )
                target = cv2.resize(
                    cv2.imread(
                        f'{data_dir}/{directory}/{os.path.splitext(filename)[0]}_AN.png'
                    ),
                    self.processed_dim
                )

                # mask = np.zeros((448, 576))
                # for idx, __ in enumerate(target):
                    # for jdx, __ in enumerate(target[idx]):
                        # if np.array_equal(target[idx, jdx], [0, 0, 255]):
                            # mask[idx, jdx] = 1
                        # elif np.array_equal(target[idx, jdx], [0, 255, 255]):
                            # mask[idx, jdx] = 2

                plt.imshow(mask)
                plt.show()
                cv2.imshow('test', mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                quit()

                target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
                lower_y = np.array([22, 93, 0], dtype='uint8')
                upper_y = np.array([44, 255, 255], dtype='uint8')
                mask_yellow = cv2.inRange(target, lower_y, upper_y) / 255
                lower_r = np.array([45, 183, 188], dtype='uint8')
                upper_r = np.array([179, 255, 255], dtype='uint8')
                mask_red = cv2.inRange(target, lower_r, upper_r) / 255
                for idx, __ in enumerate(mask_red):
                    for jdx, __ in enumerate(mask_red[idx]):
                        if mask_yellow[idx, jdx] == 1:
                            mask_red[idx, jdx] = 2
                mask = mask_red

                vbb = int(re.search(r'\d+', os.path.splitext(filename)[0].split('_')[-1]).group())
                data.append([feature, mask, vbb])
        return data

    def process_raw(self):
        """Generating target images from raw data"""
        data = self.read_raw(self.data_dir)
        # train, test = train_test_split(data, test_size=self.config['test_size'])
        train = data[:len(data) // 2]
        test = data[len(data) // 2:]

        for train_test in tuple(zip([train, test], ['train', 'test'])):
            for idx, sample in enumerate(train_test[0]):
                cv2.imwrite(
                    f'{self.processed_dir}/{train_test[1]}/{self.data_lbls[0]}/1/{idx:03d}_{sample[2]}.png',
                    sample[0]
                )
                np.save(
                    f'{self.processed_dir}/{train_test[1]}/{self.data_lbls[1]}/1/{idx:03d}_{sample[2]}.npy',
                    sample[1]
                )

    def get_batches(self):
        return self.train_data

    def validate(self, evaluate, epoch_idx):
        print("Start validation...")

        Path(
            '{}/epoch{}'.format(self.results_dir, epoch_idx)
        ).mkdir(parents=True, exist_ok=True)

        acc = []
        for batch_idx, batch in enumerate(self.test_data):
            inp = batch['F']
            out = batch['T']
            pred_out = evaluate(inp)

            # local_acc = jaccard_score(
                # torch.argmax(out, dim=1).flatten(),
                # torch.argmax(pred_out, dim=1).flatten().cpu(),
                # average='macro'
            # )
            # print(local_acc)
            # acc.append(local_acc)

            for image_idx, image in enumerate(pred_out):
                inp_image = inp[image_idx].permute(1, 2, 0)
                inp_image = (inp_image - inp_image.min()) / (inp_image.max() - inp_image.min())
                out_image = torch.argmax(out[image_idx], dim=0)
                pred_out_image = torch.argmax(image, dim=0).cpu()

                save_idx = batch_idx * self.batch_size + image_idx

                acc.append((out_image == pred_out_image).float().mean().item() * 100.0)

                save_idx = batch_idx * self.batch_size + image_idx
                self.plot_results(
                    [inp_image, pred_out_image, out_image],
                    ['Input', 'Output', 'Target'],
                    f'{self.results_dir}/epoch{epoch_idx}/pred_{save_idx}.png'
                )

        return np.mean(acc), np.std(acc)

    def infer(self, evaluate, infer_dir):
        """Inference method"""
        Path('{}/predictions'.format(infer_dir[0])).mkdir(parents=True, exist_ok=True)

        # data = self.read_raw(f'{infer_dir[0]}/data')

        # if not any(Path(f'{infer_dir[0]}/test/{self.data_lbls[0]}/1').iterdir()):
            # for idx, sample in enumerate(data):
                # cv2.imwrite(
                    # f'{infer_dir[0]}/test/{self.data_lbls[0]}/1/{idx:03d}_{sample[2]}.png',
                    # sample[0]
                # )
                # np.save(
                    # f'{infer_dir[0]}/test/{self.data_lbls[1]}/1/{idx:03d}_{sample[2]}.npy',
                    # sample[1]
                # )

        target_available = any(Path(f'{infer_dir[0]}/test/{self.data_lbls[-1]}/1').iterdir())

        infer_dataset = WearDataset(
            f'{infer_dir[0]}/test/{self.data_lbls[0]}',
            f'{infer_dir[0]}/test/{self.data_lbls[1]}' if target_available else None
        )

        infer_data = DataLoader(infer_dataset, self.batch_size, shuffle=True, num_workers=0)

        acc = []
        for batch_idx, batch in enumerate(infer_data):
            inp = batch['F']
            out = batch['T'] if target_available else None
            pred_out = evaluate(inp)

            # local_acc = jaccard_score(
                # torch.argmax(out, dim=1).flatten(),
                # torch.argmax(pred_out, dim=1).flatten().cpu(),
                # average='macro'
            # )
            # print(local_acc)
            # acc.append(local_acc)

            for image_idx, image in enumerate(pred_out):
                inp_image = inp[image_idx].permute(1, 2, 0)
                inp_image = (inp_image - inp_image.min()) / (inp_image.max() - inp_image.min())
                pred_out_image = torch.argmax(image, dim=0).cpu()
                save_idx = batch_idx * self.batch_size + image_idx

                if out is not None:
                    out_image = torch.argmax(out[image_idx], dim=0)
                    acc.append((out_image == pred_out_image).float().mean().item() * 100.0)

                    self.plot_results(
                        [inp_image, pred_out_image, out_image],
                        ['Input', 'Output', 'Target'],
                        f'{infer_dir[0]}/predictions/pred_{save_idx}.png',
                    )
                else:
                    plt.imsave(
                        f'{infer_dir[0]}/predictions/pred_{save_idx}.png',
                        pred_out_image,
                        cmap='Greys'
                    )
        if acc:
            print(f"Accuracy: {np.mean(acc)} +- {np.std(acc)}")

    def plot_results(self, data, titles, filename):
        """Plot prediction results"""
        __, axs = plt.subplots(1, len(data), sharey=True)
        for idx, image in enumerate(data):
            axs[idx].imshow(image)
            axs[idx].set_title(titles[idx])
        plt.savefig(
            filename,
            format='png',
            dpi=600,
            bbox_inches='tight'
        )
        plt.close()
