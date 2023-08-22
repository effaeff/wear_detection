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
from torchvision.transforms import ToTensor, Normalize, Compose, functional, Resize, CenterCrop
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from torchmetrics import JaccardIndex
from torchmetrics.functional.classification import jaccard_index
from sklearn.metrics import jaccard_score
from config import PROCESSED_DIM, RESIZE_SIZE

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

        self.path_features = path_features
        self.path_target = path_target

        self.transform = Compose([
            ToTensor(),
            CenterCrop(PROCESSED_DIM[0]),
            # CenterCrop(PROCESSED_DIM)
            Resize(RESIZE_SIZE)
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = Compose([
            # CenterCrop(PROCESSED_DIM)
            CenterCrop(PROCESSED_DIM[0]),
            Resize(RESIZE_SIZE)
        ])

        self.f_names = sorted(
            os.listdir(f'{path_features}/1'), key=lambda name: int(re.search('\d+', name.split('_')[2]).group())
        )
        # self.f_names = sorted(
            # os.listdir(f'{path_features}/1'), key=lambda name: int(name.split('_')[0])
        # )
        # self.data_features = [
            # # np.array(Image.open(f'{path_features}/1/{f_name}'))
            # Image.open(f'{path_features}/1/{f_name}')
            # for f_name in f_names
        # ]
        self.t_names = sorted(
            os.listdir(f'{path_target}/1'), key=lambda name: int(re.search('\d+', name.split('_')[2]).group())
        )
        # self.t_names = sorted(
            # os.listdir(f'{path_target}/1'), key=lambda name: int(name.split('_')[0])
        # )
        # self.data_targets = None
        # if path_target is not None:
            # self.data_targets = np.array([
                # np.load(f'{path_target}/1/{t_name}')
                # for t_name in t_names
            # ])

    def __getitem__(self, index):
        features = Image.open(
            f'{self.path_features}/1/{self.f_names[index]}'
        )
        features = self.transform(features).float()

        targets = []
        if self.path_target is not None:
            targets = np.load(f'{self.path_target}/1/{self.t_names[index]}')
            targets = nn.functional.one_hot(
                torch.LongTensor(targets),
                num_classes=3
            ).permute(2, 0, 1).float()
            targets = self.target_transform(targets)
            targets = torch.argmax(targets, dim=0)

        item = {'F': features, 'T': targets}
        return item

    def __len__(self):
        return len(self.f_names) # Assume that both datasets have equal length

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
        self.output_size = self.config.get('output_size', 3)

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

    def validate(self, evaluate, epoch_idx, save_eval, verbose, save_suffix):
        if verbose:
            print("Start validation...")

        if save_eval:
            Path(
                '{}/epoch{}'.format(self.results_dir, epoch_idx)
            ).mkdir(parents=True, exist_ok=True)

        jacc = JaccardIndex(num_classes=3, task='multiclass', average=None)
        acc = []
        # for batch_idx, batch in enumerate(tqdm(self.test_data)):
        for batch_idx, batch in enumerate(self.test_data):
            inp = batch['F']
            out = batch['T']
            pred_out, pred_edges = evaluate(inp)
            # pred_out = evaluate(inp)

            for image_idx, image in enumerate(pred_out):
                inp_image = inp[image_idx].permute(1, 2, 0)
                # inp_image = (inp_image - inp_image.min()) / (inp_image.max() - inp_image.min())
                # out_image = torch.argmax(out[image_idx], dim=0)
                out_image = out[image_idx]
                pred_out_image = torch.argmax(image, dim=0).cpu()
                # pred_edges_image = pred_edges[image_idx].cpu()

                save_idx = batch_idx * self.batch_size + image_idx

                iou_sk = jaccard_score(
                    out_image.flatten(),
                    pred_out_image.flatten(),
                    labels=[0, 1, 2],
                    average=None,
                    zero_division=0
                )
                iou_tm = jaccard_index(
                    pred_out_image,
                    out_image,
                    task='multiclass',
                    num_classes=3,
                    average=None
                )
                # print(f"sk: {iou_sk}\ttm: {iou_tm}")
                # iou = jacc(pred_out_image, out_image)
                for label in range(3):
                    if label not in torch.unique(out_image):
                       iou_tm[label] = torch.nan

                acc.append(torch.nanmean(iou_tm))
                # acc.append(iou)

                if save_eval:
                    target = out_image.cpu().detach().numpy()
                    target_edges = np.zeros(target.shape)
                    for label in range(np.max(target)):
                        local_target = target.copy()
                        local_target[np.where(target==label+1)] = 0
                        local_blur = cv2.GaussianBlur((local_target*255).astype('uint8'), (5, 5), 0)
                        local_edges = cv2.Canny(local_blur, 100, 200) / 255
                        target_edges += local_edges

                    im_path = f'{self.results_dir}/epoch{epoch_idx}/pred_{save_idx:03d}'
                    Path(im_path).mkdir(parents=True, exist_ok=True)
                    self.plot_results(
                        [
                            inp_image,
                            pred_out_image,
                            out_image,
                            # pred_edges_image,
                            target_edges
                        ],
                        ['Input', 'Prediction', 'Target', 'Pred edges', 'Target edges'],
                        f'{self.results_dir}/epoch{epoch_idx}/{save_idx:03d}.png'
                    )
                    np.save(f'{im_path}/input.npy', inp_image)
                    np.save(f'{im_path}/pred.npy', pred_out_image)
                    np.save(f'{im_path}/target.npy', out_image)
                    # np.save(f'{im_path}/pred_edges.npy', pred_edges_image)
                    np.save(f'{im_path}/target_edges.npy', target_edges)

        if verbose:
            print(f"Validation error: {np.mean(acc)*100.0:.2f} % +- {np.std(acc)*100.0:.2f}")
        return np.mean(acc) * 100.0, np.std(acc) * 100.0

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
            axs[idx].imshow(image, cmap='inferno')
            axs[idx].set_title(titles[idx])
        plt.savefig(
            filename,
            format='png',
            dpi=600,
            bbox_inches='tight'
        )
        plt.close()
