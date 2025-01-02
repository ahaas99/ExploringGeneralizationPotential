"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script generates the corrupted MedMNIST-C dataset
"""

import argparse
import yaml
import torch
import torch.nn as nn
import timm
import pandas as pd
from pathlib import Path
from huggingface_hub import login
import time
import medmnist
import random
import numpy as np
import torchvision.transforms as transforms
import os
import sys
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights
from medmnist import INFO
from sklearn.neighbors import NearestNeighbors
from medmnist import INFO
from medmnistc.corruptions.registry import CORRUPTIONS_DS, DATASET_RGB
from medmnistc.utils.utils import seed_everything
from medmnist import INFO
from PIL import Image
import numpy as np
import os

from tqdm import tqdm


class DatasetManager:
    def __init__(self,
                 medmnist_path: str,
                 output_path: str,
                 random_seed: int = 0):
        """
        Class used to create the corrupted test sets.
        Speficially, it will create one `npz` file for each designed dataset-corruption.
        :param medmnist_path: Path to the medmnist datasets. Pre-download the 224 version.
                            https://medmnist.com/
        :param output_path: Path to the output folder of the `medmnistc` dataset.
                        Path convention: {output_folder} / {dataset} / {corruption}.npz
        :param random_seed: Control stochastic process and ensure reproducibility.
        """
        self.medmnist_path = medmnist_path
        self.output_path = output_path

        self.supported_datasets = [
            'bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist',
            'octmnist', 'organamnist', 'organcmnist', 'organsmnist',
            'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist'
        ]

        self.random_seed = random_seed

    def create_dataset(self, dataset_name: str, size: int):
        """
        Create the corrupted dataset(s).

        :param dataset_name: Name of the dataset to corrupt.
                             Options: {'all',
                                       'bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist',
                                       'octmnist', 'organamnist', 'organcmnist', 'organsmnist',
                                       'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist'}
                             If `all` is set, it will create create all the corrupted datasets.
        """

        # Create all the corrupted datasets
        if dataset_name == 'all':
            for ds in self.supported_datasets:
                self.create_single_dataset(dataset_name=ds, size=size)

        # Create only the chosen corrupted dataset
        else:
            dataset_name = dataset_name.lower()
            assert dataset_name in self.supported_datasets, f"Dataset not found. Please choose one among : {self.supported_datasets}"
            self.create_single_dataset(dataset_name=dataset_name, size=size)

    def create_single_dataset(self, dataset_name: str, size: int):
        """
        Generate one corrupted version for the required dataset.
        Note that we store one .npz file for each designed corruptions.

        :param dataset_name: Name of the dataset to corrupt.
        """
        print(f"=========== {dataset_name} ===========")

        rng = seed_everything(self.random_seed)

        # Get the designed corruptions
        corruptions = CORRUPTIONS_DS[dataset_name]

        # Get MedMNIST's dataset class
        #   It required the pre-download of the 224 datasets
        info = INFO[dataset_name]
        DatasetClass = getattr(__import__('medmnist', fromlist=[info['python_class']]), info['python_class'])

        dataset = DatasetClass(split="test",
                               as_rgb=True,
                               download=True,
                               transform=None,
                               size=size,
                               root=self.medmnist_path)

        dataset_path = os.path.join(self.output_path, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)

        # Create the corrupted datasets
        # NOTE: This could be computationally heavy (RAM-wise) for large datasets (e.g. TissueMNIST),
        #       as it multiply 5 times the test set.
        dataset_c, labels = [], []

        for (corruption, corruptor) in corruptions.items():

            print(f'Starting {corruption}...')

            # Load the corrupted images and relative labels into lists
            dataset_c, labels = [], []

            if corruption == "impulse_noise":
                corruptor.rng = rng  # skimage..
            ##Let corruptions out if size ==28


            # By design, we have 5 intensity levels
            for severity in range(0, 5):
                if (size == 28 and (
                        corruption == "stain_deposit" or corruption == "bubble" or corruption == "characters")):
                    break
                if size != 224 and (
                        corruption == "motion_blur" or corruption == "characters"):
                    break

                # Define corruptor method
                lam_corruption = lambda img: corruptor.apply(img, severity)

                # Iterate over the MedMNIST dataset and apply corruptions
                for img_idx in tqdm(range(len(dataset.imgs)), f'Severity {str(severity + 1).zfill(2)}'):

                    img, label = dataset.imgs[img_idx], dataset.labels[img_idx]

                    # The defined corruptions support RGB images
                    img = Image.fromarray(img).convert('RGB')
                    corrupted_img = lam_corruption(img)

                    # Convert to greyscale, if required
                    if not DATASET_RGB[dataset_name]:
                        corrupted_img = Image.fromarray(corrupted_img).convert('L')

                    np_corrupted = np.array(corrupted_img)
                    assert np.min(np_corrupted) >= 0 or np.max(
                        np_corrupted) <= 255, f"(min,max) = {(np.min(np_corrupted), np.max(np_corrupted))}"
                    assert np_corrupted.dtype == np.uint8, f"{np_corrupted.dtype}"

                    dataset_c.append(np_corrupted)
                    labels.append(label)
            if size == 28 and (corruption == "stain_deposit" or corruption == "bubble" or corruption == "characters"):
                continue
            elif size != 224 and (
                    corruption == "motion_blur" or corruption == "characters"):
                continue
            else:
                filepath = os.path.join(dataset_path, f'{corruption}.npz')
                np.savez_compressed(filepath, test_images=dataset_c, test_labels=labels)
import numpy as np
from medmnistc.dataset import CorruptedMedMNIST
from medmnistc.eval import Evaluator
from medmnistc.corruptions.registry import CORRUPTIONS_DS
# Import custom modules
from utily import (calculate_passed_time, seed_worker, extract_embeddings, extract_embeddings_alexnet,
                   extract_embeddings_densenet, knn_majority_vote, get_ACC, get_AUC, get_ACC_kNN,get_Cohen, get_Balanced_ACC, get_AUC_kNN, get_Cohen_kNN, get_Precision)
from torch.utils.data import Dataset
from backbones import get_backbone
if __name__ == '__main__':
    """
    Parses command-line arguments and uses the DatasetManager to create a MedMNIST-C dataset.

    Arguments:
        --medmnist_path (str): Path to the MedMNIST dataset.
        --medmnistc_path (str): Path where the MedMNIST-C dataset should be stored.
        --size (int): Desired image size for the dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--medmnist_path", required=True, type=str, help="Path to the medmnist dataset.")
    parser.add_argument("--medmnistc_path", required=True, type=str, help="Path where medmnist-c should be store")
    parser.add_argument("--size", required=True, type=int, help="Image size")
    args = parser.parse_args()

    
    ds_manager = DatasetManager(medmnist_path=args.medmnist_path, output_path=args.medmnist_path)
    ds_manager.create_dataset(dataset_name="all", size=args.size)

