"""Trainer class for wear segmentation"""

import numpy as np
import cv2

from pytorchutils.globals import torch, DEVICE
from pytorchutils.basic_trainer import BasicTrainer
from tqdm import tqdm


class Trainer(BasicTrainer):
    """Wrapper class for training routine"""
    def __init__(self, config, model, dataprocessor):
        BasicTrainer.__init__(self, config, model, dataprocessor)

    def learn_from_epoch(self, epoch_idx):
        """Training method"""
        epoch_loss = 0
        try:
            batches = self.get_batches_fn()
        except AttributeError:
            print(
                "Error: No nb_batches_fn defined in preprocessor. "
                "This attribute is required by the training routine."
            )
        pbar = tqdm(batches, desc=f'Epoch {epoch_idx}', unit='batch')
        for batch_idx, batch in enumerate(pbar):
            inp = batch['F']
            out = batch['T']

            # pred_out, pred_edges = self.model(inp.to(DEVICE))
            pred_out = self.model(inp.to(DEVICE))

            pred_out = torch.sigmoid(pred_out)
            # pred_edges = torch.sigmoid(pred_edges)

            # out_border = torch.empty(
                # (out.size()[0], out.size()[-2], out.size()[-1])
            # )
            # for idx, image in enumerate(out):
                # target = np.argmax(image.cpu().detach().numpy(), axis=0)

                # target_edges = np.zeros(target.shape)
                # for label in range(np.max(target)):
                    # local_target = target.copy()
                    # local_target[np.where(target==label+1)] = 0
                    # local_blur = cv2.GaussianBlur((local_target*255).astype('uint8'), (5, 5), 0)
                    # local_edges = cv2.Canny(local_blur, 100, 200) / 255
                    # target_edges += local_edges

                # out_border[idx] = torch.from_numpy(target_edges).float()

            # batch_loss = self.loss(
                # pred_out,
                # out.to(DEVICE)
            # ) + self.loss(
                # pred_edges,
                # out_border.to(DEVICE)
            # )
            batch_loss = self.loss(pred_out, out.to(DEVICE))

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()
            pbar.set_postfix(batch_loss=batch_loss.item(), epoch_loss=epoch_loss/(batch_idx+1))
        epoch_loss /= len(batches)

        return epoch_loss

    def evaluate(self, inp):
        """Prediction and error estimation for given input and output"""
        with torch.no_grad():
            # Switch to PyTorch's evaluation mode.
            # Some layers, which are used for regularization, e.g., dropout or batch norm layers,
            # behave differently, i.e., are turnd off, in evaluation mode
            # to prevent influencing the prediction accuracy.
            if isinstance(self.model, (list, np.ndarray)):
                for idx, __ in enumerate(self.model):
                    self.model[idx].eval()
            else:
                self.model.eval()

            # pred_out, pred_edges = self.model(inp.to(DEVICE))
            pred_out = self.model(inp.to(DEVICE))
            pred_out = torch.sigmoid(pred_out)
            # pred_edges = torch.sigmoid(pred_edges)

            return pred_out#, pred_edges
