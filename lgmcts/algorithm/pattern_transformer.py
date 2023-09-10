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
import pytorch_lightning as pl

from lgmcts.algorithm.transformer import *

# -1 is the beginning, -2 is the end, -3 for unknown values


def createPointDataset(seq_len):
    merged_result = []
    for i in range(seq_len):
        # Define the dimensions of your 2D array
        rows, cols = 100, 200

        # Create a 2D array filled with zeros
        true_locations = np.zeros((rows, cols))

        # Define the number of true 1s you want to place
        num_true_ones = 3

        # Randomly select locations for the true 1s

        # Place the true 1s in the array
        true_locations[25][125] = 1
        true_locations[25][75] = 1
        true_locations[75][75] = 1

        # Set the desired standard deviation
        desired_std_deviation = 5.0

        results = []
        temp_array = np.zeros((rows, cols))
        np.random.seed(None)
        random_number = int(round(np.random.normal(0, desired_std_deviation)))
        np.random.seed(None)
        random_number_2 = int(round(np.random.normal(0, desired_std_deviation)))
        random_dir = np.random.randint(0, 4)

        newarray = None
        if random_dir == 0:
            newarray = [25-random_number, 125-random_number_2]
        elif random_dir == 1:
            newarray = [25-random_number, 125+random_number_2]
        elif random_dir == 2:
            newarray = [25+random_number, 125-random_number_2]
        elif random_dir == 3:
            newarray = [25+random_number, 125+random_number_2]

        results.extend(newarray)

        current_time = int(time.time())
        np.random.seed(None)
        random_number = int(round(np.random.normal(0, desired_std_deviation)))
        np.random.seed(None)
        random_number_2 = int(round(np.random.normal(0, desired_std_deviation)))
        random_dir = np.random.randint(0, 4)
        if random_dir == 0:
            results.extend([25-random_number, 75-random_number_2])
        elif random_dir == 1:
            results.extend([25-random_number, 75+random_number_2])
        elif random_dir == 2:
            results.extend([25+random_number, 75-random_number_2])
        elif random_dir == 3:
            results.extend([25+random_number, 75+random_number_2])

        current_time = int(time.time())
        np.random.seed(None)
        random_number = int(round(np.random.normal(0, desired_std_deviation)))
        np.random.seed(None)
        random_number_2 = int(round(np.random.normal(0, desired_std_deviation)))
        random_dir = np.random.randint(0, 4)
        if random_dir == 0:
            results.extend([75-random_number, 75-random_number_2])
        elif random_dir == 1:
            results.extend([75-random_number, 75+random_number_2])
        elif random_dir == 2:
            results.extend([75+random_number, 75-random_number_2])
        elif random_dir == 3:
            results.extend([75+random_number, 75+random_number_2])
        results.extend([0, 0, 0, 0, 0, 0])
        merged_result.append(results)
    return merged_result


class ReverseDataset(data.Dataset):
    def __init__(self, seq_len):
        super().__init__()
        self.num_categories = 12
        self.seq_len = seq_len

        output = torch.tensor(createPointDataset(seq_len))

        self.size = len(output)
        self.data = torch.tensor(output)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]
        labels = inp_data.clone().detach()

        random_number = random.randint(0, 2)
        if random_number == 0:
            positions_to_replace = [2, 3, 4, 5]
            for i in positions_to_replace:
                inp_data[i] = -1
        elif random_number == 1:
            positions_to_replace = [0, 1, 4, 5]
            for i in positions_to_replace:
                inp_data[i] = -1
        elif random_number == 2:
            positions_to_replace = [0, 1, 2, 3]
            for i in positions_to_replace:
                inp_data[i] = -1

        return inp_data.to(torch.float32), labels.to(torch.float32)


def train_reverse(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "ReverseTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=3,
                         gradient_clip_val=3)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    model = ReversePredictor(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set

    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
    return model, result
