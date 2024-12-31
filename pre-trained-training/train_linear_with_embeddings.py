"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script trains a model on a specified dataset of the MedMNIST+ collection and saves the best performing model.
"""

# Import packages
import argparse
import yaml
import torch
import torch.nn as nn
import timm
import time
import medmnist
import random
import numpy as np
import torchvision.transforms as transforms
from huggingface_hub import login
import wandb
import pandas as pd
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from torchvision.models import alexnet, AlexNet_Weights
from medmnist import INFO
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
# Import custom modules
from utils import calculate_passed_time, seed_worker, get_ACC_kNN, get_AUC_kNN, get_ACC, get_AUC, get_Balanced_ACC, get_Cohen, get_Precision


def train_only_head(config: dict, train_loader: DataLoader, val_loader: DataLoader, model_to_use):
    """
    Train a model on the specified dataset and save the best performing model.

    :param config: Dictionary containing the parameters and hyperparameters.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param model_to_use: model that should be used to train.
    """
    architecture_name = ""
    if config['architecture'] == 'hf_hub:prov-gigapath/prov-gigapath':
        architecture_name = "prov"
    elif config['architecture'] == "hf_hub:timm/vit_base_patch14_dinov2.lvd142m":
        architecture_name = "dinov2"
    elif config['architecture'] == "vit_base_patch16_224.dino":
        architecture_name = "dino"
    elif architecture == "alexnet":
        architecture_name = "alexnet"
    else:
        architecture_name = "uni"
    # Start code
    start_time = time.time()
    print("\tStart training ...")

    run = wandb.init(
        # set the wandb project where this run will be logged
        project=architecture_name,

        # track hyperparameters and run metadata
        config={
            "learning_rate": config["learning_rate"],
            "architecture": config["architecture"],
            "dataset": config["dataset"],
            "epochs": config["epochs"],
            "img_size": config["img_size"]
        }
    )
    # Initialize the model for the given training procedure
    print("\tInitialize the model for the given training procedure ...")


    model = model_to_use#.get_classifier()
    for param in model.parameters():
        param.requires_grad = False

    if config['architecture'] == 'alexnet':
        for param in model.classifier[6].parameters():
            param.requires_grad = True
    else:
        for param in model.get_classifier().parameters():
            param.requires_grad = True



    # Move the model to the available device
    model = model.to(config['device'])

    # Create the optimizer and the learning rate scheduler
    print("\tCreate the optimizer and the learning rate scheduler ...")
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    #scheduler = CosineLRScheduler(optimizer, t_initial=config['epochs'], cycle_limit=1, t_in_epochs=True)

    # Define the loss function
    print("\tDefine the loss function ...")
    if config['task'] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss().to(config['device'])
        prediction = nn.Sigmoid()
    else:
        criterion = nn.CrossEntropyLoss().to(config['device'])
        prediction = nn.Softmax(dim=1)

    # Create variables to store the best performing model
    print("\tInitialize helper variables ...")
    best_loss, best_epoch = np.inf, 0
    best_model = deepcopy(model)
    epochs_no_improve = 0  # Counter for epochs without improvement
    n_epochs_stop = 5  # Number of epochs to wait before stopping

    # Training loop
    print(f"\tRun the training for {config['epochs']} epochs ...")
    print(f"\tRun the training for {config['epochs']} epochs ...")
    for epoch in range(config['epochs']):
        start_time_epoch = time.time()  # Stop the time
        print(f"\t\tEpoch {epoch} of {config['epochs']}:")

        # Training
        print(f"\t\t\t Train:")
        model.train()
        train_loss = 0
        y_true_train, y_pred_train = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])
        for images, labels in tqdm(train_loader):
            # Map the data to the available device
            images = images.to(config['device'])


            if config['task'] == 'multi-label, binary-class':
                labels = labels.to(torch.float32).to(config['device'])
            else:
                labels = torch.squeeze(labels, 1).long().to(config['device'])

            # Run the forward pass
            outputs = model(images)


            # Compute the loss and perform backpropagation
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            outputs_deatached = outputs.clone().detach()
            # Update the weights
            optimizer.step()
            optimizer.zero_grad()
            if config['task'] == 'multi-label, binary-class':
                outputs_deatached = prediction(outputs_deatached).to(config['device'])
            else:
                outputs_deatached = prediction(outputs_deatached).to(config['device'])
                labels = labels.float().resize_(len(labels), 1)

            y_true_train = torch.cat((y_true_train, deepcopy(labels)), 0)
            y_pred_train = torch.cat((y_pred_train, deepcopy(outputs_deatached)), 0)

        ACC_Train = get_ACC(y_true_train.cpu().numpy(), y_pred_train.cpu().numpy(), config['task'])
        AUC_Train = get_AUC(y_true_train.cpu().numpy(), y_pred_train.cpu().numpy(), config['task'])
        Bal_Acc_Train = get_Balanced_ACC(y_true_train.cpu().numpy(), y_pred_train.cpu().numpy(), config['task'])
        Co_Train = get_Cohen(y_true_train.cpu().numpy(), y_pred_train.cpu().numpy(), config['task'])
        Prec_Train = get_Precision(y_true_train.cpu().numpy(), y_pred_train.cpu().numpy(), config['task'])
        # Update the learning rate
        #scheduler.step(epoch=epoch)

        # Evaluation
        print(f"\t\t\t Evaluate:")
        model.eval()
        val_loss = 0
        y_true, y_pred = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                # Map the data to the available device
                images = images.to(config['device'])
                outputs = model(images)

                # Run the forward pass
                if config['task'] == 'multi-label, binary-class':
                    labels = labels.to(torch.float32).to(config['device'])
                    loss = criterion(outputs, labels)
                    outputs = prediction(outputs).to(config['device'])

                else:
                    labels = torch.squeeze(labels, 1).long().to(config['device'])
                    loss = criterion(outputs, labels)
                    outputs = prediction(outputs).to(config['device'])
                    labels = labels.float().resize_(len(labels), 1)


                # Store the current loss
                val_loss += loss.item()

                # Store the labels and predictions
                y_true = torch.cat((y_true, deepcopy(labels)), 0)
                y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)


        # Calculate the metrics
        ACC = get_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        AUC = get_AUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Bal_Acc = get_Balanced_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Co = get_Cohen(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        Prec = get_Precision(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        # Print the loss values and send them to wandb
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"\t\t\tTrain Loss: {train_loss}")
        print(f"\t\t\tVal Loss: {val_loss}")
        print(f"\t\t\tACC: {ACC}")
        print(f"\t\t\tAUC: {AUC}")
        wandb.log({"accuracy": ACC, "auc": AUC, "accuracy_train": ACC_Train, "auc_train": AUC_Train, "val_loss": val_loss, "train_loss": train_loss, "bal_acc": Bal_Acc, "cohen": Co, "prec": Prec, "bal_acc_train": Bal_Acc_Train, "cohen_train": Co_Train, "prec_train": Prec_Train})
        # Store the current best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1  # Increment the counter

        print(f"\t\t\tCurrent best Val Loss: {best_loss}")
        print(f"\t\t\tCurrent best Epoch: {best_epoch}")

        # Check for early stopping
        if epochs_no_improve == n_epochs_stop:
            print("\tEarly stopping!")
            break

        # Stop the time for the epoch
        end_time_epoch = time.time()
        hours_epoch, minutes_epoch, seconds_epoch = calculate_passed_time(start_time_epoch, end_time_epoch)
        print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch, seconds_epoch))

    print(f"\tSave the trained model ...")
    #Path(config['output_path']).mkdir(parents=True, exist_ok=True)

    save_name = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_headonly_schedueler_{architecture_name}_s{config['seed']}"

    torch.save(model.state_dict(), f"{save_name}_final.pth")
    torch.save(best_model.state_dict(), f"{save_name}_best.pth")
    print(best_model.state_dict())
    print(f"\tFinished training.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time for training: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))
    run.finish()

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
    wandb.login()
    if config['architecture'] == 'alexnet':
        torch.backends.cudnn.benchmark = True  # Enable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = False  # Disable cudnn to be deterministic
        torch.use_deterministic_algorithms(False)  # Disable only deterministic algorithms
    else:
        torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = True  # Enable cudnn to be deterministic
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
        for img_size in [28, 64, 128, 224]:
    # Extract the dataset and its metadata
            info = INFO[dataset]
            config['task'], config['in_channel'], config['num_classes'] = info['task'], info['n_channels'], len(info['label'])
            DataClass = getattr(medmnist, info['python_class'])
            #for architecture in ['alexnet','hf_hub:prov-gigapath/prov-gigapath', "hf_hub:timm/vit_base_patch14_dinov2.lvd142m", "vit_base_patch16_224.dino", "hf-hub:MahmoodLab/uni"]:
            architecture=config["architecture"]
            print(f"\t\t\t ... for {architecture}...")
            access_token = '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'
                # Create the model
            if architecture == 'alexnet':
                model = alexnet(weights=AlexNet_Weights.DEFAULT)
                #model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
                # Remove the classifier
                model.classifier[6] = nn.Linear(4096, config['num_classes'])
            elif architecture == 'hf-hub:MahmoodLab/uni':
                login(access_token)
                model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5,
                                              dynamic_img_size=True, num_classes=num_classes)
            elif architecture == 'hf_hub:prov-gigapath/prov-gigapath':
                login(access_token)
                model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, num_classes=num_classes)
            else:
                model = timm.create_model(architecture, pretrained=True, num_classes=num_classes)
            # Create the data transforms and normalize with imagenet statistics
            if architecture == 'alexnet':
                mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Use ImageNet statistics
            else:
                mean, std = model.default_cfg['mean'], model.default_cfg['std']
            if architecture == 'hf-hub:MahmoodLab/uni':
                if img_size == 28:
                    total_padding = 4
                else:
                    total_padding = 0
            elif architecture == 'hf_hub:timm/vit_base_patch14_dinov2.lvd142m':
                total_padding = max(0, 518 - img_size)
            else :
                total_padding = max(0, 224 - img_size)

            padding_left, padding_top = total_padding // 2, total_padding // 2
            padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top


            data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0, padding_mode='constant')  # Pad the image to 224x224
            ])
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

            #train_data = DataFromDict(data_train)
            #val_data = DataFromDict(data_val)

            #train_loader = DataLoader(data_train, batch_size=config['batch_size'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
            #print(train_loader)    # test_loader = DataLoader(data_test, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g
                # Create the datasets
            train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=True, size=img_size)
            val_dataset = DataClass(split='val', transform=data_transform, download=True, as_rgb=True, size=img_size)

                # Create the dataloaders
            #print(data_train)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4
                                 , worker_init_fn=seed_worker, generator=g)
            #val_loaderttest = DataLoader(val_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4
            #                        , worker_init_fn=seed_worker, generator=g)

            config["dataset"] = dataset
            config["img_size"] = img_size
            config["architecture"] = architecture

            train_only_head(config, train_loader, val_loader, model_to_use=model)
