# Standard libraries
import os
import time
import numpy as np
import random
import math
import json
from functools import partial
import scipy.stats as stats
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import lgmcts.utils.misc_utils as utils
from lgmcts.algorithm.transformer import *

from lgmcts.components.patterns import PATTERN_DICT


def createPatternData(rng, seq_len, sample_len=3, pattern_type="line"):
    pattern_type = "line"
    datas = []
    img_size = (100, 100)
    rng = np.random.Generator(np.random.PCG64(12345))
    for i in range(seq_len):
        prior = PATTERN_DICT[pattern_type].gen_prior(img_size=img_size, rng=rng)[0]
        samples = utils.sample_distribution(prob=prior, rng=rng, n_samples=sample_len)
        samples = samples.astype(np.float32)
        samples[:, 0] = samples[:, 0] / float(img_size[0])
        samples[:, 1] = samples[:, 1] / float(img_size[1])
        # scale to [0, 1]
        datas.append(samples.flatten())
    return np.vstack(datas)


def pose_tokensize(pose: np.ndarray, resolution: float = 0.01):
    """Tokenize 2d pose"""
    num_tokens = int(1 / resolution)
    pose = np.clip(pose, 0, 1)
    pose = (pose * num_tokens).astype(np.int32)
    return pose[:, 0] * num_tokens + pose[:, 1]


class PatternDataset(data.Dataset):
    def __init__(self, seq_len, device="cpu"):
        super().__init__()
        self.num_categories = 12
        self.sample_len = 3
        self.seq_len = seq_len
        self.pattern_type = "line"
        self.resolution = 0.01
        self.rng = np.random.Generator(np.random.PCG64(12345))

        # create data
        self.data = createPatternData(self.rng, self.seq_len, self.sample_len, self.pattern_type)
        self.size = self.data.shape[0]
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        inp_data = inp_data.reshape(self.sample_len, 2)
        labels = np.copy(inp_data)

        # apply a random mask
        num_masked = 1
        mask_idx = self.rng.choice(self.sample_len, num_masked, replace=False)
        inp_data[mask_idx] = 0

        return torch.from_numpy(inp_data).to(self.device), torch.from_numpy(labels).to(self.device)


class PatternPredictor(TransformerPredictor):

    def _calculate_loss(self, batch, mode="train"):
        inp_data, labels = batch  # inp_data: (batch_size, sample_len*2), labels: (batch_size, sample_len*2)

        preds = self.forward(inp_data, add_positional_encoding=True)

        # use a l2 loss
        loss = F.mse_loss(preds, labels)

        # calculate accuracy
        batch_size = inp_data.shape[0]
        preds = preds.detach().cpu().numpy().reshape(batch_size, -1)
        labels = labels.detach().cpu().numpy().reshape(batch_size, -1)
        acc = np.mean(np.linalg.norm(preds - labels, axis=-1) < 0.1)

        # Logging
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


if __name__ == "__main__":
    root_dir = "/home/robot-learning/Projects/LGMCTS-D/output/pattern_transformer"
    device = "cpu"
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=50,
                         gradient_clip_val=3)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    total_number = 10000
    train_dataset, val_dataset, test_dataset = random_split(
        PatternDataset(total_number, device=device), [int(total_number * 0.8), int(total_number * 0.1), int(total_number * 0.1)]
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = PatternPredictor(input_dim=2,
                             model_dim=32,
                             num_heads=1,
                             output_dim=2,
                             num_layers=1,
                             dropout=0.0,
                             lr=5e-4,
                             max_iters=100,
                             warmup=50)
    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
