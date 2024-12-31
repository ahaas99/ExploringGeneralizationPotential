"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script evaluates a model on a specified dataset of the MedMNIST+ collection.
"""

# Import packages
import argparse
import pickle

import yaml
import torch
import torch.nn as nn
import timm
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
import pandas as pd
from pathlib import Path

# Import custom modules
from utils import (calculate_passed_time, seed_worker, extract_embeddings, extract_embeddings_alexnet,
                   extract_embeddings_densenet, knn_majority_vote, get_ACC, get_AUC, get_ACC_kNN,get_Balanced_ACC_kNN, get_Cohen_kNN, get_AUC_kNN)




def evaluate_with_embeddings(config: dict, support_set: dict, data_set: dict, k: int, dataset: str):
    """
    Evaluate a model on the specified dataset.

    :param config: Dictionary containing the parameters and hyperparameters.
    :param support_set: Dictionary representing the training set, used for training the kNN.
    :param data_set: Test set, used for evaluation.
    :param k: Number of k Nearest Neighbours.
    :param dataset: Name of the dataset.
    """

    # Start code
    start_time = time.time()

    # Load the trained model
    print("\tLoad the trained model ...")

    # Extract the dataset and its metadata
    info = INFO[dataset]
    config['task'], config['in_channel'], config['num_classes'] = info['task'], info['n_channels'], len(info['label'])
    architecture_name = ""
    if config['architecture'] == 'hf_hub:prov-gigapath/prov-gigapath':
        architecture_name = "prov"
    elif config['architecture'] == "hf_hub:timm/vit_base_patch14_dinov2.lvd142m":
        architecture_name = "dinov2"
    elif config['architecture'] == "vit_base_patch16_224.dino":
        architecture_name = "dino"
    else:
        architecture_name = "uni"
    if config['training_procedure'] == 'kNN':

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(support_set['embeddings'])
        filename = f"/mnt/data/modelsalex/models/{architecture_name}/knn/{config['dataset']}_{config['img_size']}.sav"
        pickle.dump(nbrs, open(filename, 'wb'))
    # Run the Evaluation
    print(f"\tRun the evaluation ...")
    y_true, y_pred = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])

    with torch.no_grad():
        for x in range(len(data_set["embeddings"])):
    #   Map the data to the available device
            #images, labels = images.to(config['device']), labels.to(torch.float32).to(config['device'])

            if config['training_procedure'] == 'kNN':
                outputs = data_set["embeddings"][x]
                outputs = outputs.reshape(1, -1)
                #outputs = outputs.detach().cpu().numpy()
                outputs = knn_majority_vote(nbrs, outputs, support_set['labels'], config['task'])
                outputs = outputs.to(config['device'])

            if config['task'] == "multi-label, binary-class":
                #print(np.array(data_set["labels"][x]))
                true_prediction = torch.from_numpy(np.array(data_set["labels"][x]))
            else:
                true_prediction = torch.from_numpy(np.array(data_set["labels"][x]))
            true_prediction = true_prediction.reshape(true_prediction.shape[0], -1)
            true_prediction = true_prediction.to(config['device'])
            y_true = torch.cat((y_true, deepcopy(true_prediction)), 0)
            y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)
        #for embeddings, labels in tqdm(test_loader):
            # Map the data to the available device
            #embeddings, labels = embeddings.to(config['device']), labels.to(torch.float32).to(config['device'])
            #outputs = embeddings.reshape(embeddings.shape[0], -1)
            #outputs = outputs.detach().cpu().numpy()
            #outputs = knn_majority_vote(nbrs, outputs, support_set['labels'], config['task'])
            #outputs = outputs.to(config['device'])



             #Store the labels and predictions
            #y_true = torch.cat((y_true, deepcopy(labels)), 0)
           # y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)
        # Calculate the metrics
        if config['training_procedure'] == 'kNN':
            if config['task'] == "multi-label, binary-class":
                y_true = y_true.reshape(22433, -1)
            ACC = get_ACC_kNN(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
            AUC = 0.0  # AUC cannot be calculated for the kNN approach
            Bal_Acc = get_Balanced_ACC_kNN(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
            Co = get_Cohen_kNN(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])  # AUC cannot be calculated for the kNN approach

        else:
            ACC = get_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
            AUC = get_AUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])

        # Print the loss values and send them to wandb
        print(f"\t\t\tACC: {ACC}")
        print(f"\t\t\tAUC: {AUC}")
        return ACC,Bal_Acc,Co

    print(f"\tFinished evaluation.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time for evaluation: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--dataset", required=False, type=str, help="Which dataset to use.")
    parser.add_argument("--img_size", required=False, type=int, help="Which image size to use.")
    parser.add_argument("--training_procedure", required=False, type=str, help="Which training procedure to use.")
    parser.add_argument("--architecture", required=False, type=str, help="Which architecture to use.")
    parser.add_argument("--k", required=False, type=int, help="Number of nearest neighbors to use.")
    parser.add_argument("--seed", required=False, type=int, help="Which seed was used during training.")
    parser.add_argument("--output_path", required=False, type=str,
                        help="Path to the output folder.")
    parser.add_argument("--output_path_embeddings", required=False, type=str,
                        help="Path to the output folder of the embeddings.")
    parser.add_argument("--output_path_acc", required=False, type=str,
                        help="Path to the output folder of the metrics.")
    args = parser.parse_args()
    config_file = args.config_file

    # Load the parameters and hyperparameters of the configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Adapt to the command line arguments
    if args.dataset:
        config['dataset'] = args.dataset

    if args.img_size:
        config['img_size'] = args.img_size

    if args.training_procedure:
        config['training_procedure'] = args.training_procedure

    if args.architecture:
        config['architecture'] = args.architecture

    if args.k:
        config['k'] = args.k

    if args.seed:
        config['seed'] = args.seed
     
    if args.output_path:
        config['output_path'] = args.output_path

    if args.output_path_embeddings:
        config['output_path_embeddings'] = args.output_path_embeddings

    if args.output_path_acc:
        config['output_path_acc'] = args.output_path_acc

    # Seed the training and data loading so both become deterministic
    if config['architecture'] == 'alexnet':
        torch.backends.cudnn.benchmark = True  # Enable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = False  # Disable cudnn to be deterministic
        torch.use_deterministic_algorithms(False)  # Disable only deterministic algorithms

    else:
        torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = True  # Enable cudnn to be deterministic

        if config['architecture'] == 'samvit_base_patch16':
            torch.use_deterministic_algorithms(True, warn_only=True)  # Enable only deterministic algorithms

        else:
            torch.use_deterministic_algorithms(True)  # Enable only deterministic algorithms

    torch.manual_seed(config['seed'])  # Seed the pytorch RNG for all devices (both CPU and CUDA)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    g = torch.Generator()
    g.manual_seed(config['seed'])


    #for dataset in ['bloodmnist', 'breastmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist',
    #            'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist', 'chestmnist']:
    for dataset in ['chestmnist']:
        print(f"\t... for {dataset}...")
        df = pd.DataFrame(columns=["dataset", "img_size", "Acc", "Bal_Acc", "Co"])
        for img_size in [28, 64, 128, 224]:

            print(f"\t\t... for size {img_size}...")
          if img_size == 28:
                filename = Path(config["output_path_embeddings"]) / f"{dataset}_embeddings.npz"
            else:
                filename = Path(config["output_path_embeddings"]) / f"{dataset}_{img_size}_embeddings.npz"


            data = np.load(filename, allow_pickle=True)
            #data["arr_0"].item()["train"]["embeddings"]
            data_train = data["arr_0"].item()["train"]
            data_test = data["arr_0"].item()["test"]
            config["dataset"] = dataset
            config["img_size"] = img_size
            #train_loader = DataLoader(data_train, batch_size=config['batch_size'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
            #test_loader = DataLoader(data_test, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
           # print(test_loader)
            acc = 0
            anzahlk = 5
            k = 11
            acc, bal_acc, co = evaluate_with_embeddings(config, data_train, data_test, k, dataset)
            d = {'dataset': [dataset], 'img_size': img_size, "Acc":[acc], "Bal_Acc": bal_acc, "Co":co}
            dfsupport = pd.DataFrame(data=d)
            df = pd.concat([df, dfsupport])

            filename = Path(config["output_path_acc") / f"{dataset}_acc.csv"

            df.to_csv(filename, index=False)
            print(f"Best = {acc}, Knoten = {anzahlk}")








    # Run the training
    #evaluate(config, train_loader, test_loader)
