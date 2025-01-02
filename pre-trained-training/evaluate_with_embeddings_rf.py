"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script trains and evaluates a model on all datasets of the MedMNIST+ collection.
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
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_blobs
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Import custom modules
from utils import (calculate_passed_time, seed_worker, extract_embeddings, extract_embeddings_alexnet,
                   extract_embeddings_densenet, knn_majority_vote, get_ACC, get_AUC, get_ACC_kNN,get_Balanced_ACC_kNN, get_Cohen_kNN, get_AUC_kNN)




def evaluate_with_embeddings_rf(config: dict, support_set: dict, validation_set: dict, data_set: dict, dataset: str):
    """
    Evaluates a Random Forest model using precomputed embeddings on a specified dataset.

    Parameters:
        config (dict): Dictionary containing parameters for the Random Forest model and evaluation process.
        support_set (dict): Dictionary representing the training set.
        validation_set (dict): Dictionary representing the validation set, used to assess performance during or after training.
        data_set (dict): Dictionary containing the test dataset to evaluate performance.
        dataset (str): Name of the dataset to be evaluated, used for logging or selecting specific configurations.
    """
    # Start code
    start_time = time.time()

    # Load the trained model
    print("\tLoad the trained model ...")

    # Extract the dataset and its metadata
    info = INFO[dataset]
    config['task'], config['in_channel'], config['num_classes'] = info['task'], info['n_channels'], len(info['label'])


    #Create the classifier
    rf = RandomForestClassifier(
        max_depth=7,
        n_estimators=50,
        min_samples_split=5,
        min_samples_leaf=3
    )



    # Fit the data to the OvR classifier
    ovr_classifier = rf.fit(support_set["embeddings"], support_set["labels"])
    filename = f"{config["output_path"]}/{config["architecture_name"]}/rf/{config['dataset']}_{config['img_size']}.sav"
    pickle.dump(ovr_classifier, open(filename, 'wb'))


    # Run the Evaluation
    print(f"\tRun the evaluation ...")
    y_true, y_pred = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])

    with torch.no_grad():
        #evaluate the model on train, validation and test split
        outputs_train = ovr_classifier.predict(support_set["embeddings"])
        true_prediction_train = support_set["labels"]
        outputs_train = np.asarray(outputs_train)
        outputs_train = outputs_train.reshape(outputs_train.shape[0], -1)
        y_pred_prob_train = ovr_classifier.predict_proba(support_set["embeddings"])
        true_prediction_train = true_prediction_train.reshape(true_prediction_train.shape[0], -1)
        acc_train = get_ACC_kNN(true_prediction_train, outputs_train, config['task'])
        bal_acc_train = get_Balanced_ACC_kNN(true_prediction_train, outputs_train, config['task'])
        co_train = get_Cohen_kNN(true_prediction_train, outputs_train, config['task'])
        auc_train = get_AUC_kNN(true_prediction_train, y_pred_prob_train, config['task'])
        # Print the loss values and send them to wandb
        print(f"\t\t\tACC: {acc_train}")

        outputs_val = ovr_classifier.predict(validation_set["embeddings"])
        true_prediction_val = validation_set["labels"]
        outputs = np.asarray(outputs_val)
        outputs = outputs.reshape(outputs.shape[0], -1)
        y_pred_prob_val = ovr_classifier.predict_proba(validation_set["embeddings"])
        true_prediction_val = true_prediction_val.reshape(true_prediction_val.shape[0], -1)
        acc_val = get_ACC_kNN(true_prediction_val, outputs, config['task'])
        bal_acc_val = get_Balanced_ACC_kNN(true_prediction_val, outputs_val, config['task'])
        co_val = get_Cohen_kNN(true_prediction_val, outputs_val, config['task'])
        auc_val = get_AUC_kNN(true_prediction_val, y_pred_prob_val, config['task'])
        # Print the loss values and send them to wandb
        print(f"\t\t\tACC: {acc_val}")

        outputs = ovr_classifier.predict(data_set["embeddings"])
        true_prediction = data_set["labels"]
        outputs = np.asarray(outputs)
        outputs = outputs.reshape(outputs.shape[0], -1)
        y_pred_prob = ovr_classifier.predict_proba(data_set["embeddings"])
        true_prediction = true_prediction.reshape(true_prediction.shape[0], -1)
        acc = get_ACC_kNN(true_prediction, outputs, config['task'])
        bal_acc = get_Balanced_ACC_kNN(true_prediction, outputs, config['task'])
        co = get_Cohen_kNN(true_prediction, outputs, config['task'])
        auc = get_AUC_kNN(true_prediction, y_pred_prob, config['task'])
        print(f"\t\t\tACC: {acc}")
        return acc_train, acc_val, acc, bal_acc_train, bal_acc_val, bal_acc, auc_train, auc_val, auc, co_train, co_val, co

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

    for dataset in ['bloodmnist', 'breastmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist',
                    'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist', 'chestmnist']:
        # for dataset in ['bloodmnist']:
        print(f"\t... for {dataset}...")
        df = pd.DataFrame(columns=["dataset", "img_size", "Acc_Test", "Acc_Val", "Acc_Train", "Bal_Acc", "Bal_Acc_Val",
                                   "Bal_Acc_Train", "AUC", "AUC_Val", "AUC_Train", "CO", "CO_Val", "CO_Train"])
        for img_size in [28, 64, 128, 224]:
            print(f"\t\t... for size {img_size}...")
            if img_size == 28:
                filename = Path(config["output_path_embeddings"]) / f"{dataset}_embeddings.npz"
            else:
                filename = Path(config["output_path_embeddings"]) / f"{dataset}_{img_size}_embeddings.npz

            data = np.load(filename, allow_pickle=True)
            #Load the embeddings 
            data_train = data["arr_0"].item()["train"]
            data_validation = data["arr_0"].item()["val"]
            data_test = data["arr_0"].item()["test"]
            config["dataset"] = dataset
            config["img_size"] = img_size

             # train and evaluate the model  
            acc_train, acc_val, acc, bal_acc_train, bal_acc_val, bal_acc, auc_train, auc_val, auc, co_train, co_val, co = evaluate_with_embeddings_rf(
                config, data_train, data_validation, data_test,
                dataset)
            d = {'dataset': [dataset], 'img_size': img_size, "Acc_Test": [acc], "Acc_Val": [acc_val],
                 "Acc_Train": [acc_train], "Bal_Acc": [bal_acc], "Bal_Acc_Val": [bal_acc_val],
                 "Bal_Acc_Train": [bal_acc_train], "AUC": [auc], "AUC_Val": [auc_val], "AUC_Train": [auc_train],
                 "CO": [co], "CO_Val": [co_val], "CO_Train": [co_train]}

            #add the metrics and save them
            dfsupport = pd.DataFrame(data=d)
            df = pd.concat([df, dfsupport])

            filename = Path(config["output_path_acc"]) / f"{dataset}_acc.csv"


            df.to_csv(filename, index=False)
            print(f"= {acc} ")








    # Run the training
    #evaluate(config, train_loader, test_loader)
