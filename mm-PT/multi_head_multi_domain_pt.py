import copy

import pandas as pd
import torch
import time
import torch.nn as nn
import numpy as np
import wandb
from medmnist import INFO
from copy import deepcopy
from timm.optim import AdamW
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler
from utils import calculate_passed_time, get_ACC, get_AUC, get_Balanced_ACC, \
    get_Cohen, get_Precision
from backbones import get_backbone

from typing import Callable, Optional
from typing_extensions import deprecated

from torch import Tensor
from torch.nn import _reduction as _Reduction, functional as F

#Custom Loss if you want to weight the loss depending on the dataset
class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(CustomBCEWithLogitsLoss, self).__init__()

    def forward(self, inputs, targets, multiplier):
        return multiplier * F.binary_cross_entropy_with_logits(
            inputs,
            targets
        )


#Custom Loss if you want to weight the loss depending on the dataset
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor, multiplier) -> Tensor:
        return multiplier * F.cross_entropy(
            input,
            target,
        )


# function to generate a dictionary holding the dataloaders for the 12 specified 2D datasets of medmnist+
def generate_heads(dataset_names, num_features, device):
    print(f"Generating heads for the {len(dataset_names)} datasets {dataset_names}")
    head_dict = {}  # dictionary that holds the different heads corresponding to the datasets
    # dictionary that holds the best heads corresponding to the datasets
    task_dict = {}  # dictionary that holds the different tasks corresponding to the datasets
    # fills dictionary with strings describing the task corresponding to the dataset
    for dataset_name in dataset_names:
        task_string = INFO[dataset_name]['task']
        task_dict[dataset_name] = task_string
        num_classes = len(INFO[dataset_name]['label'])
        print(f"Initializing head for {dataset_name} with the task of {task_string} and thus {num_classes} Classes")
        head_dict[dataset_name] = torch.nn.Linear(in_features=num_features, out_features=num_classes, device=device)
    return head_dict, task_dict


def multi_head_multi_domain_training(config: dict, loader_dict):
    """
    Train a multi-head-multi-domain model on a collection of datasets

    :param config: Dictionary containing the parameters and hyperparameters.
    :param loader_dict: Holds two dictionaries with a collection of dataloaders inside them, one dictionary for the
    collection of dataloaders for the training split, the other dictionary for the collection of dataloaders for the
    validation split
    """
    head_dict_best = {}

    # Start code
    start_time = time.time()
    print("\tStart training ...")
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=config["architecture_name"] + "_multi_head",

        # track hyperparameters and run metadata
        config={
            'learning_rate': config['learning_rate'],
            'architecture': config['architecture'],
            'architecture_name': config['architecture_name'],
            'epochs': config['epochs'],
            'img_size': config['img_size'],
            'batch_size': config['batch_size'],
            'num_workers': config['num_workers'],
            'device': config['device']
        }
    )

    architecture = config['architecture']
    learning_rate = config['learning_rate']
    backbone_name = config['architecture_name']
    device = config['device']
    dataset_names = list(INFO)[:12]

    # initialise backbone
    backbone, num_features = get_backbone(backbone_name=backbone_name, architecture=architecture, num_classes=1000,
                                          pretrained=True)
    if list(loader_dict['train_loader_dict'].keys()) == list(loader_dict['val_loader_dict'].keys()):
        dataset_names = list(loader_dict['train_loader_dict'].keys())  # crucial list for indexing, holds dataset names

    # instantiate dictionaries holding the different heads for each dataset and their tasks, all indexable by means of
    # the dataset name
    head_dict, task_dict = generate_heads(dataset_names=dataset_names, num_features=num_features, device=device)
    model = backbone
    # Move the model to the available device
    model = model.to(device)
    #for param in model.parameters():
    #   param.requires_grad = True
    # Create the optimizer
    print("\tCreate the optimizer")
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Get lengths of training dataloaders
    loader_len_train = {}  # holds the lengths of the dataloader for each dataset
    sum_loader_length_train = 0  # holds total length of all the dataloaders in a given split combined
    for dataset_name, train_loader in loader_dict['train_loader_dict'].items():
        loader_len_train[dataset_name] = len(train_loader)
        sum_loader_length_train += len(train_loader)
    loader_len_val = {}  # holds the lengths of the dataloader for each dataset
    sum_loader_length_val = 0  # holds total length of all the dataloaders in a given split combined
    for dataset_name, validation_loader in loader_dict['val_loader_dict'].items():
        loader_len_val[dataset_name] = len(validation_loader)
        sum_loader_length_val += len(validation_loader)
    # Compute initial probabilities of picking dataloader to get a batch from
    probability_dict = {}
    for dataset_name in dataset_names:
        probability_dict[dataset_name] = loader_len_train[dataset_name] / sum_loader_length_train
    probability_dict_val = {}
    for dataset_name in dataset_names:
        probability_dict_val[dataset_name] = loader_len_val[dataset_name] / sum_loader_length_val

    # Create variables to store the best performing model
    print("\tInitialize helper variables ...")
    best_loss, best_epoch = np.inf, 0
    best_acc = 0
    best_model = deepcopy(model)
    epochs_no_improve = 0  # Counter for epochs without improvement
    n_epochs_stop = 5  # Number of epochs to wait before stopping
    scheduler = CosineLRScheduler(optimizer, t_initial=config['epochs'], cycle_limit=1, t_in_epochs=True)
    # Training loop
    dfalle = pd.DataFrame()
    print(f"\tRun the training for {config['epochs']} epochs ...")
    for epoch in range(config['epochs']):
        start_time_epoch = time.time()  # Stop the time
        print(f"\t\tEpoch {epoch + 1} of {config['epochs']}:")

        # Reset the data iterator for each dataset
        data_iterartor = {}
        for dataset_name, train_loader in loader_dict['train_loader_dict'].items():
            data_iterartor[dataset_name] = iter(
                train_loader)

        print(data_iterartor)

        # reset length dictionary and thus probabilities each epoch
        len_dict = copy.deepcopy(loader_len_train)
        running_probability_dict = copy.deepcopy(probability_dict)
        print(f"Probability dictionary: {running_probability_dict}")

        # reset training accuracy for epoch each epoch
        train_acc_list = []

        # Training
        print(f"\t\t\t Train:")
        model.train()
        train_loss = 0
        # y_true_train, y_pred_train = torch.tensor([]).to(device), torch.tensor([]).to(device)
        # y_true_train_multi_label, y_pred_train_multi_label = torch.empty((1, 14)).to(device), torch.empty((1, 14)).to(
        #    device)

        running_loader_length_sum = sum_loader_length_train
        y_true_train_dict, y_pred_train_dict = {}, {}

        for dataset_name in dataset_names:
            y_true_train_dict[dataset_name] = torch.tensor([]).to(device)
            y_pred_train_dict[dataset_name] = torch.tensor([]).to(device)

        for i in range(sum_loader_length_train):
            # get probabilities from dictionary as list
            prob_list = list(running_probability_dict.values())
            # choose random dataset according to weighted probabilities
            random_dataset = np.random.choice(dataset_names, p=prob_list)
            # print(f"random_dataset: {random_dataset}")
            # get batch from dataset which was randomly chosen

            images, labels = next(data_iterartor[random_dataset])


            # adjust probabilities for later use
            # print(f"len_dict = {len_dict}")
            if len_dict[random_dataset] > 0:
                len_dict[random_dataset] -= 1
            else:
                raise ValueError("length and probability of dataloader can't be less than 0")

            # update probability dictionary
            running_loader_length_sum -= 1
            # would become zero and thus cause a division by zero at the very last batch
            if running_loader_length_sum >= 1:
                for dataset_name in dataset_names:
                    running_probability_dict[dataset_name] = len_dict[dataset_name] / running_loader_length_sum

            # Map the data to the available device
            images = images.to(device)

            # set loss functions (has to be done each iteration, so per batch, as task will change)
            # use a weighted loss if config['weighted_loss']==True
            if task_dict[random_dataset] == "multi-label, binary-class":
                if config['weighted_loss']:
                    criterion = CustomBCEWithLogitsLoss().to(device)
                else:
                    criterion = nn.BCEWithLogitsLoss().to(device)
                prediction = nn.Sigmoid()
            else:
                if config['weighted_loss']:
                    criterion = CustomCrossEntropyLoss().to(device)
                else:
                    criterion = nn.CrossEntropyLoss().to(device)
                prediction = nn.Softmax(dim=1)

            if task_dict[random_dataset] == 'multi-label, binary-class':
                labels = labels.to(torch.float32).to(device)
            else:
                labels = torch.squeeze(labels, 1).long().to(device)

            # Swap Head for the Mode
            model.head = head_dict[random_dataset]

            # Run the forward pass
            outputs = model(images)
            # Compute the loss and perform backpropagation
            if config['weighted_loss']:
                loss = criterion(outputs, labels, multiplier=loader_len_train['breastmnist']/loader_len_train[random_dataset])
            else:
                loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            outputs_detached = outputs.clone().detach()
            # Update the weights
            optimizer.step()
            optimizer.zero_grad()
            if task_dict[random_dataset] == 'multi-label, binary-class':
                outputs_detached = prediction(outputs_detached).to(device)
            else:
                outputs_detached = prediction(outputs_detached).to(device)
                labels = labels.float().resize_(len(labels), 1)

            # update head
            head_dict[random_dataset] = model.get_classifier()

            y_true_train_dict[random_dataset] = torch.cat((y_true_train_dict[random_dataset], deepcopy(labels)), 0)
            y_pred_train_dict[random_dataset] = torch.cat(
                (y_pred_train_dict[random_dataset], deepcopy(outputs_detached)), 0)
            # get accuracy for single batch, as it relies on the task which might change each batch
            train_acc_list.append(
                get_ACC(labels.cpu().numpy(),
                        outputs_detached.cpu().numpy(), task_dict[random_dataset]))

        # average all the batch-wise accuracies to obtain "global" accuracy per epoch
        ACC_Train = np.mean(train_acc_list)
        Acc_TrainPartitioned = 0
        for dataset_name in dataset_names:
            Acc = get_ACC(y_true_train_dict[dataset_name].cpu().numpy(), y_pred_train_dict[dataset_name].cpu().numpy(),
                          task_dict[dataset_name])
            Acc_TrainPartitioned = Acc_TrainPartitioned + Acc
        Acc_TrainPartitioned=Acc_TrainPartitioned/12
        df = pd.DataFrame()
        
        #compute the metrics and safe in a DataFrame
        for dataset_name in dataset_names:
            Acc = get_ACC(y_true_train_dict[dataset_name].cpu().numpy(), y_pred_train_dict[dataset_name].cpu().numpy(), task_dict[dataset_name])
            column1 = "Acc_" + str(dataset_name)+ "_Train"
            Auc = get_AUC(y_true_train_dict[dataset_name].cpu().numpy(), y_pred_train_dict[dataset_name].cpu().numpy(),
                          task_dict[dataset_name])
            column2 = "Auc_" + str(dataset_name) + "_Train"
            Bal_Acc = get_Balanced_ACC(y_true_train_dict[dataset_name].cpu().numpy(), y_pred_train_dict[dataset_name].cpu().numpy(),
                          task_dict[dataset_name])
            column3 = "Bal_Acc" + str(dataset_name) + "_Train"
            Co = get_Cohen(y_true_train_dict[dataset_name].cpu().numpy(), y_pred_train_dict[dataset_name].cpu().numpy(),
                          task_dict[dataset_name])
            column4 = "Co_" + str(dataset_name) + "_Train"
            Prec = get_Precision(y_true_train_dict[dataset_name].cpu().numpy(), y_pred_train_dict[dataset_name].cpu().numpy(),
                           task_dict[dataset_name])
            column5 = "Prec_" + str(dataset_name) + "_Train"
            d = {column1: [Acc], column2: [Auc], column3:[Bal_Acc], column4:[Co], column5: [Prec]}
            df1 = pd.DataFrame(data= d)
            if df.empty:
                df = df1.copy()
            else:
                df = df.join(df1)


        # Update the learning rate
        scheduler.step(epoch=epoch)

        # Get lengths of training dataloaders
        loader_len_val = {}  # holds the lengths of the dataloader for each dataset
        sum_loader_length_val = 0  # holds total length of all the dataloaders in a given split combined
        for dataset_name, val_loader in loader_dict['val_loader_dict'].items():
            loader_len_val[dataset_name] = len(val_loader)
            sum_loader_length_val += len(val_loader)

        # Evaluation
        print(f"\t\t\t Evaluate:")
        model.eval()
        val_loss = 0
        y_true_val_dict, y_pred_val_dict = {}, {}
        for dataset_name in dataset_names:
            y_true_val_dict[dataset_name] = torch.tensor([]).to(device)
            y_pred_val_dict[dataset_name] = torch.tensor([]).to(device)

        # reset validation accuracy for epoch each epoch
        val_acc_list = []

        with torch.no_grad():
            # iterating over all validation dataloaders
            for dataset_name, single_val_loader in loader_dict['val_loader_dict'].items():
                # Swap Head for the Model
                model.head = head_dict[dataset_name]
                # use a weighted loss if config['weighted_loss']==True
                if task_dict[dataset_name] == "multi-label, binary-class":
                    if config['weighted_loss']:
                        criterion = CustomBCEWithLogitsLoss().to(device)
                    else:
                        criterion = nn.BCEWithLogitsLoss().to(device)
                    prediction = nn.Sigmoid()
                else:
                    if config['weighted_loss']:
                        criterion = CustomCrossEntropyLoss().to(device)
                    else:
                        criterion = nn.CrossEntropyLoss().to(device)
                    prediction = nn.Softmax(dim=1)

                # iterating over all the batches of a single dataloader
                for images, labels in tqdm(single_val_loader):
                    # Map the data to the available device
                    images = images.to(device)
                    outputs = model(images)
                    # Run the forward pass
                    if task_dict[dataset_name] == 'multi-label, binary-class':
                        labels = labels.to(torch.float32).to(device)
                        if config['weighted_loss']:
                            loss = criterion(outputs, labels, multiplier=loader_len_val["breastmnist"]/loader_len_val[dataset_name])
                        else:
                            loss = criterion(outputs, labels)
                        outputs = prediction(outputs)

                    else:
                        labels = torch.squeeze(labels, 1).long().to(device)
                        if config['weighted_loss']:
                            loss = criterion(outputs, labels, multiplier=loader_len_val["breastmnist"]/loader_len_val[dataset_name])
                        else:
                            loss = criterion(outputs, labels)
                        outputs = prediction(outputs)
                        labels = labels.float().resize_(len(labels), 1)

                    # Store the current loss
                    val_loss += loss.item()

                    # Store the labels and predictions
                    y_true_val_dict[dataset_name] = torch.cat((y_true_val_dict[dataset_name], deepcopy(labels)),
                                                              0)
                    y_pred_val_dict[dataset_name] = torch.cat((y_pred_val_dict[dataset_name], deepcopy(outputs)), 0)

                # average all the batch-wise accuracies to obtain "global" accuracy per epoch
                    val_acc_list.append(
                    get_ACC(labels.cpu().numpy(),
                            outputs.cpu().numpy(), task_dict[dataset_name]))

        # Calculate the metrics
        ACC = np.mean(val_acc_list)
        Acc_ValPartitioned = 0
        for dataset_name in dataset_names:
            Acc = get_ACC(y_true_val_dict[dataset_name].cpu().numpy(), y_pred_val_dict[dataset_name].cpu().numpy(),
                          task_dict[dataset_name])
            Acc_ValPartitioned = Acc_ValPartitioned+Acc
        Acc_ValPartitioned=Acc_ValPartitioned/12
        for dataset_name in dataset_names:
            Acc = get_ACC(y_true_val_dict[dataset_name].cpu().numpy(), y_pred_val_dict[dataset_name].cpu().numpy(),
                          task_dict[dataset_name])
            column1 = "Acc_" + str(dataset_name) + "_Val"

            Auc = get_AUC(y_true_val_dict[dataset_name].cpu().numpy(), y_pred_val_dict[dataset_name].cpu().numpy(),
                          task_dict[dataset_name])
            column2 = "Auc_" + str(dataset_name) + "_Val"

            Bal_Acc = get_Balanced_ACC(y_true_val_dict[dataset_name].cpu().numpy(),
                                       y_pred_val_dict[dataset_name].cpu().numpy(),
                                       task_dict[dataset_name])
            column3 = "Bal_Acc" + str(dataset_name) + "_Val"

            Co = get_Cohen(y_true_val_dict[dataset_name].cpu().numpy(), y_pred_val_dict[dataset_name].cpu().numpy(),
                           task_dict[dataset_name])
            column4 = "Co_" + str(dataset_name) + "_Val"

            Prec = get_Precision(y_true_val_dict[dataset_name].cpu().numpy(),
                                 y_pred_val_dict[dataset_name].cpu().numpy(),
                                 task_dict[dataset_name])
            column5 = "Prec_" + str(dataset_name) + "_Val"
            d = {column1: [Acc], column2: [Auc], column3: [Bal_Acc], column4: [Co], column5: [Prec]}
            df1 = pd.DataFrame(data=d)
            df = df.join(df1)
        if dfalle.empty:
            dfalle=df.copy()
        else:
            dfalle = pd.concat([dfalle, df])
        
        #safe the metrics for training and validation
        dfalle.to_csv("accuracies" + config["architecture_name"]+ "_" + str(config["img_size"]) + ".csv", index=False)


        print(f"\t\t\tTrain Loss: {train_loss}")
        print(f"\t\t\tVal Loss: {val_loss}")
        print(f"\t\t\tACC: {ACC}")
        #strore the parameters in wandb
        wandb.log(
            {"accuracy": ACC, "accuracytrainpartitioned": Acc_TrainPartitioned,
             "accuracypartitioned": Acc_ValPartitioned, "accuracy_train": ACC_Train, "val_loss": val_loss,
             "train_loss": train_loss})

        if best_loss > val_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
            head_dict_best = deepcopy(head_dict)
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
    #safe the heads and the backbones, the backbone gets saved with a newly initialized head 
    classifier = torch.nn.Linear(in_features=num_features, out_features=1000, device=config["device"])
    save_name = f"{config['output_path']}/{config['architecture_name']}/{config['img_size']}/s{config['seed']}"
    model.head = classifier
    best_model.head = classifier
    torch.save(model.state_dict(), f"{save_name}_backbone_final.pth")
    torch.save(best_model.state_dict(), f"{save_name}_backbone_best.pth")
    for dataset_name in dataset_names:
        torch.save(head_dict_best[dataset_name].state_dict(), f"{save_name}_{dataset_name}_final.pth")
        torch.save(head_dict[dataset_name].state_dict(), f"{save_name}_{dataset_name}_best.pth")

