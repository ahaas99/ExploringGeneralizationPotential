"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script evaluates a model on a specified dataset of the MedMNIST+ collection.
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
from medmnist import INFO
# Import custom modules
from utily import (calculate_passed_time, seed_worker, extract_embeddings, extract_embeddings_alexnet,
                   extract_embeddings_densenet, knn_majority_vote, get_ACC, get_AUC, get_ACC_kNN,get_Cohen, get_Balanced_ACC, get_AUC_kNN, get_Cohen_kNN, get_Precision)
from torch.utils.data import Dataset
from backbones import get_backbone
sys.path.insert(0, '/accuracies/')
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

class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def evaluate(config: dict, dataset,test_loader: DataLoader):
    """
    Evaluate a model on the specified dataset.

    :param config: Dictionary containing the parameters and hyperparameters.
    :param train_loader: DataLoader for the training set.
    :param test_loader: DataLoader for the test set.
    """

    # Start code
    start_time = time.time()

    # Load the trained model
    print("\tLoad the trained model ...")
    architecture_name = ""
    if architecture == 'hf_hub:prov-gigapath/prov-gigapath':
        architecture_name = "prov"
    elif architecture == "hf_hub:timm/vit_base_patch14_dinov2.lvd142m":
        architecture_name = "dinov2"
    elif architecture == "vit_base_patch16_224.dino":
        architecture_name = "dino"
    elif architecture == "alexnet":
        architecture_name = "alexnet"
    else:
        architecture_name = "uni"
    #print(model.state_dict())



    task_string = INFO[dataset]['task']

    num_classes = len(INFO[dataset]['label'])
    print(f"Initializing head for {dataset} with the task of {task_string} and thus {num_classes} Classes")
    model, num_features = get_backbone(backbone_name=architecture_name, architecture=architecture, num_classes=1000,
                                          pretrained=True)
    checkpoint_file = f"{config['output_path']}/{config['architecture_name']}/{config['img_size']}/s{config['seed']}"
    checkpoint = torch.load(f"{checkpoint_file}_backbone_best.pth", map_location='cpu')

    model.head = torch.nn.Linear(in_features=num_features, out_features=1000, device=config["device"])
    model.load_state_dict(checkpoint)  # , strict=False)
    classifier = torch.nn.Linear(in_features=num_features, out_features=num_classes, device=config["device"])
    checkpoint_file = f"{config['output_path']}/{config['architecture_name']}/{config['img_size']}/s{config['seed']}"
    checkpoint = torch.load(f"{checkpoint_file}_{dataset}_best.pth", map_location='cpu')
    classifier.load_state_dict(checkpoint)  # , strict=False)
    print(model.head)

    model.head = classifier
    print(model.head)
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
        for images, labels in tqdm(test_loader):
            # Map the data to the available device
            images, labels = images.to(config['device']), labels.to(torch.float32).to(config['device'])
            #print(labels)

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
    return  ACC_Test, Bal_Acc_Test,  AUC_Test,  Co_Test, Prec_Test

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
    parser.add_argument("--output_path", required=False, type=str, default='/mnt/data/modelsalex/models/multihead_neu',
                        help="Path to the output folder.")
    parser.add_argument("--embeddings_path", required=False, type=str, default='embeddings/',
                        help="Path to the output folder.")
    parser.add_argument("--output_path_acc", required=False, type=str, default='accuracies/',
                        help="Path to the output folder.")
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

    if args.embeddings_path:
        config['embeddings_path'] = args.embeddings_path

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

    for dataset in ['breastmnist', 'bloodmnist', 'chestmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist',
                    'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist']:

        print(f"\t... for {dataset}...")
        # Extract the dataset and its metadata
        info = INFO[dataset]
        task, in_channel, num_classes = info['task'], info['n_channels'], len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        # Iterate over all image sizes
        for img_size in [28, 64,128,224]:
            df = pd.DataFrame(columns=["dataset", "img_size", "Acc_Test", "Bal_Acc", "AUC", "CO", "Prec"])
            # Extract the dataset and its metadata
            info = INFO[dataset]
            config['task'], config['in_channel'], config['num_classes'] = info['task'], info['n_channels'], len(
                info['label'])
            DataClass = getattr(medmnist, info['python_class'])
            # for architecture in ['hf_hub:prov-gigapath/prov-gigapath', "hf_hub:timm/vit_base_patch14_dinov2.lvd142m", "vit_base_patch16_224.dino", "hf-hub:MahmoodLab/uni"]:
            architecture = config["architecture"]
            print(f"\t\t\t ... for {architecture}...")
            access_token = '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'
            # Create the model
            if architecture == 'alexnet':
                model = alexnet(weights=AlexNet_Weights.DEFAULT)
                model.classifier[6] = nn.Linear(4096, config['num_classes'])
                #model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Remove the classifier
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

            architecture_name = ""
            if architecture == 'hf_hub:prov-gigapath/prov-gigapath':
                architecture_name = "prov"
            elif architecture == "hf_hub:timm/vit_base_patch14_dinov2.lvd142m":
                architecture_name = "dinov2"
            elif architecture == "vit_base_patch16_224.dino":
                architecture_name = "dino"
            elif architecture == "alexnet":
                architecture_name = "alexnet"
            else:
                architecture_name = "uni"
            if img_size == 28:
                filename = Path(args.embeddings_path) / f"{architecture_name}/{dataset}_embeddings.npz"
            else:
                filename = Path(args.embeddings_path) / f"{architecture_name}/{dataset}_{img_size}_embeddings.npz"
            #print(filename)
            #data = np.load(filename, allow_pickle=True)
            # data["arr_0"].item()["train"]["embeddings"]
            #data_train = data["arr_0"].item()["train"]
            #data_val = data["arr_0"].item()["val"]
            #data_test = data["arr_0"].item()["val"]
            if architecture == 'alexnet':
                mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Use ImageNet statistics
            else:
                mean, std = model.default_cfg['mean'], model.default_cfg['std']

            if architecture == 'hf_hub:timm/vit_base_patch14_dinov2.lvd142m':
                total_padding = max(0, 518 - img_size)
            else :
                total_padding = max(0, 224 - img_size)
            padding_left, padding_top = total_padding // 2, total_padding // 2
            padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top

            data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                                   padding_mode='constant')  # Pad the image to 224x224
            ])
            #train_data = DataFromDict(data_train)
            #val_data = DataFromDict(data_val)
            #test_data = DataFromDict(data_test)
            #train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=True,
            #                          size=img_size)
            #val_dataset = DataClass(split='val', transform=data_transform, download=True, as_rgb=True, size=img_size)
            test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=True, size=img_size)
            # Create the dataloaders
            # print(data_train)
            #train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
            #                          worker_init_fn=seed_worker, generator=g)
            #val_loader = DataLoader(val_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4
            #                        , worker_init_fn=seed_worker, generator=g)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4
                                         , worker_init_fn=seed_worker, generator=g)

            config["dataset"] = dataset
            config["img_size"] = img_size
            config["architecture"] = architecture
            acc,  bal_acc,  auc, co, prec = evaluate(config,dataset=dataset,test_loader=test_loader)
            d = {'dataset': [dataset], 'img_size': img_size, "Acc_Test": [acc], "Bal_Acc": [bal_acc],  "AUC": [auc],
                 "CO": [co],"Prec": [prec] }

            dfsupport = pd.DataFrame(data=d)
            df = pd.concat([df, dfsupport])

            filename = Path(args.output_path_acc) / f"{architecture_name}/head/{img_size}/{dataset}_acc.csv"

            df.to_csv(filename, index=False)
    # Run the training
    #evaluate(config, train_loader, val_loader, test_loader, model)