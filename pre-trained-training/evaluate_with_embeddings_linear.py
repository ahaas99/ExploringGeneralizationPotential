"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script evaluates a model on all datasets of the MedMNIST+ collection.
"""

# Import packages
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

# Import custom modules
from utils import (calculate_passed_time, seed_worker, extract_embeddings, extract_embeddings_alexnet,
                   extract_embeddings_densenet, knn_majority_vote, get_ACC, get_AUC, get_ACC_kNN,get_Cohen, get_Balanced_ACC, get_AUC_kNN, get_Cohen_kNN, get_Precision)
from torch.utils.data import Dataset

sys.path.insert(0, '/embeddings/')
sys.path.insert(0, '/models/')

class DataFromDict(Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict
        self.input_keys = list(input_dict.keys())

    def __len__(self):
        return len(self.input_dict['embeddings'])

    def __getitem__(self, idx):
        item = self.input_dict['embeddings'][idx]
        label = self.input_dict['labels'][idx]
        return item, label
def evaluate(config: dict, train_loader: DataLoader, val_loader:DataLoader,test_loader: DataLoader, model):
    """
    Evaluate a model on the specified dataset.

    :param config: Dictionary containing the parameters and hyperparameters and the used dataset.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param test_loader: DataLoader for the test set.
    :param model: the model that should be trained.
    """

    # Start code
    start_time = time.time()

    # Load the trained model
    print("\tLoad the trained model ...")

        architecture_name = "uni"
    #Train only the head of the model
    model = model.get_classifier()
    checkpoint_file = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_headonly_{config["architecture_name"]}_s{config['seed']}_best.pth"
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint)
    if config['task'] == "multi-label, binary-class":
        prediction = nn.Sigmoid()
    else:
        prediction = nn.Softmax(dim=1)
    # Move the model to the available device
    model = model.to(config['device'])
    model.requires_grad_(False)
    model.eval()

    y_true, y_pred = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            # Map the data to the available device
            images, labels = images.to(config['device']), labels.to(torch.float32).to(config['device'])

            outputs = model(images)
            outputs = prediction(outputs)


            # Store the labels and predictions
            y_true = torch.cat((y_true, deepcopy(labels)), 0)
            y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)

        ACC_Train = get_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        AUC_Train = get_AUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Bal_Acc_Train = get_Balanced_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Co_Train = get_Cohen(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Prec_Train = get_Precision(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
    y_true, y_pred = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            # Map the data to the available device
            images, labels = images.to(config['device']), labels.to(torch.float32).to(config['device'])
            print(labels)

            outputs = model(images)
            outputs = prediction(outputs)

            # Store the labels and predictions
            y_true = torch.cat((y_true, deepcopy(labels)), 0)
            y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)

        ACC_Val = get_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        AUC_Val = get_AUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Bal_Acc_Val = get_Balanced_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Co_Val = get_Cohen(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Prec_Val = get_Precision(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
    y_true, y_pred = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            # Map the data to the available device
            images, labels = images.to(config['device']), labels.to(torch.float32).to(config['device'])
            print(labels)

            outputs = model(images)
            outputs = prediction(outputs)

            # Store the labels and predictions
            y_true = torch.cat((y_true, deepcopy(labels)), 0)
            y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)

        ACC_Test = get_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        AUC_Test = get_AUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Bal_Acc_Test = get_Balanced_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Co_Test = get_Cohen(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Prec_Test = get_Precision(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
    print(f"\tFinished evaluation.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time for evaluation: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))
    return ACC_Train, ACC_Val, ACC_Test, Bal_Acc_Train, Bal_Acc_Val, Bal_Acc_Test, AUC_Train, AUC_Val, AUC_Test, Co_Train, Co_Val, Co_Test, Prec_Train, Prec_Val, Prec_Test

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
        config['output_path_acc'] = args.output_path_accs

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
    # iterate over every dataset
    for dataset in ['breastmnist', 'bloodmnist', 'chestmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist',
                    'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist']:

        print(f"\t... for {dataset}...")
        # Extract the dataset and its metadata
        info = INFO[dataset]
        task, in_channel, num_classes = info['task'], info['n_channels'], len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        df = pd.DataFrame(columns=["dataset", "img_size", "Acc_Test", "Acc_Val", "Acc_Train", "Bal_Acc", "Bal_Acc_Val",
                                   "Bal_Acc_Train", "AUC", "AUC_Val", "AUC_Train", "CO", "CO_Val", "CO_Train", "Prec", "Prec_Val", "Prec_Train"])
        # Iterate over all image sizes
        for img_size in [28, 64, 128, 224]:
            # Extract the dataset and its metadata
            info = INFO[dataset]
            config['task'], config['in_channel'], config['num_classes'] = info['task'], info['n_channels'], len(
                info['label'])
            DataClass = getattr(medmnist, info['python_class'])
            architecture = config["architecture"]
            print(f"\t\t\t ... for {architecture}...")
            access_token = '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'
            # Create the model
            if architecture == 'alexnet':
                model = alexnet(weights=AlexNet_Weights.DEFAULT)
                model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Remove the classifier
            elif architecture == 'hf-hub:MahmoodLab/uni':
                login(access_token)
                model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5,
                                          dynamic_img_size=True, num_classes=num_classes)
            elif architecture == 'hf_hub:prov-gigapath/prov-gigapath':
                login(access_token)
                model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True,
                                          num_classes=num_classes)
            else:
                model = timm.create_model(architecture, pretrained=True, num_classes=num_classes)
            # Create the data transforms and normalize with imagenet statistics

        
            if img_size == 28:
                filename = Path(config["output_path_embeddings"]) / f"{dataset}_embeddings.npz"
            else:
                filename = Path(config["output_path_embeddings"]) / f"{dataset}_{img_size}_embeddings.npz"
            #Load the embeddings   
            data = np.load(filename, allow_pickle=True)
            # data["arr_0"].item()["train"]["embeddings"]
            data_train = data["arr_0"].item()["train"]
            data_val = data["arr_0"].item()["val"]
            data_test = data["arr_0"].item()["test"]

            train_data = DataFromDict(data_train)
            val_data = DataFromDict(data_val)
            test_data = DataFromDict(data_test)

            # Create the dataloaders
            
            train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                                      worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_data, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4
                                    , worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_data, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4
                                         , worker_init_fn=seed_worker, generator=g)
          
            config["dataset"] = dataset
            config["img_size"] = img_size
            config["architecture"] = architecture
            # train and evaluate the model
            acc_train, acc_val, acc, bal_acc_train, bal_acc_val, bal_acc, auc_train, auc_val, auc, co_train, co_val, co, prec_train, prec_val, prec = evaluate(config, train_loader, val_loader, test_loader, model)
            d = {'dataset': [dataset], 'img_size': img_size, "Acc_Test": [acc], "Acc_Val": [acc_val],
                 "Acc_Train": [acc_train], "Bal_Acc": [bal_acc], "Bal_Acc_Val": [bal_acc_val],
                 "Bal_Acc_Train": [bal_acc_train], "AUC": [auc], "AUC_Val": [auc_val], "AUC_Train": [auc_train],
                 "CO": [co], "CO_Val": [co_val], "CO_Train": [co_train],"Prec": [prec], "Prec_Val":[prec_val], "Prec_Train":[prec_train] }

            #add the metrics and save them  
            dfsupport = pd.DataFrame(data=d)
            df = pd.concat([df, dfsupport])

            filename = Path(config["output_path_acc"]) / f"{dataset}_acc.csv"

            df.to_csv(filename, index=False)
    
