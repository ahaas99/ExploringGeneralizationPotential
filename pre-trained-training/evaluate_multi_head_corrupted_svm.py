"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script evaluates a model on all datasets of MedMNIST-C
"""

# Import packages
import argparse
import pickle

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
from medmnistc.dataset import CorruptedMedMNIST
# Import custom modules
from utils import (calculate_passed_time, seed_worker, extract_embeddings, extract_embeddings_alexnet,
                   extract_embeddings_densenet, knn_majority_vote, get_ACC, get_AUC,get_Precision_kNN, get_ACC_kNN,get_Cohen, get_Balanced_ACC, get_AUC_kNN,get_Balanced_ACC_kNN, get_Cohen_kNN, get_Precision)

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

sys.path.insert(0, '/accuracies/')
sys.path.insert(0, '/models/')
class CorruptedMedMNIST(Dataset):
    def __init__(self,
                 dataset_name: str,
                 corruption: str,
                 norm_mean: list = [0.5],
                 norm_std: list = [0.5],
                 padding: int = 0,
                 root: str = None,
                 as_rgb: bool = True,
                 mmap_mode: str = None):
        """
        Dataset class of CorruptedMedMNIST

        :param dataset_name: Name of the reference medmnist dataset.
        :param corruption: Name of the desired corruption.
        :param norm_mean: Normalization mean.
        :param norm_std: Normalization standard deviation.
        :param root: Root path of the generated corrupted data.
        :param as_rgb: Flag for RGB of Greyscale data.
        :param mmap_mode: Memory mapping of the file: {None, ‘r+’, ‘r’, ‘w+’, ‘c’}.
                          If not None, then memory-map the file, using the given mode
                          (see numpy.memmap for a detailed description of the modes).
                          Memory mapping is especially useful for accessing small
                          fragments of large files without reading the entire file into memory.
                          src: https://numpy.org/doc/stable/reference/generated/numpy.load.html

        This dataset class was greatly inspired from the MedMNIST APIs:
            https://github.com/MedMNIST/MedMNIST
        """

        super(CorruptedMedMNIST, self).__init__()

        self.dataset_name = dataset_name
        self.corruption = corruption
        self.root = root
        self.as_rgb = as_rgb

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        if not os.path.exists(os.path.join(self.root, self.dataset_name, f"{corruption}.npz")):
            print(os.path.join(self.root, self.dataset_name, f"{corruption}.npz"))
            raise RuntimeError(
                "Dataset not found."
            )

        npz_file = np.load(
            os.path.join(self.root, self.dataset_name, f"{corruption}.npz"),
            mmap_mode=mmap_mode,
        )

        self.imgs = npz_file["test_images"]
        self.labels = npz_file["test_labels"]
        padding_left, padding_top = padding // 2, padding // 2
        padding_right, padding_bottom = padding - padding_left, padding - padding_top
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
            transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                           padding_mode='constant')
        ])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

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




def evaluate(config: dict, dataset, test_loader: DataLoader, model):
    """
    Evaluates a model on the specified dataset.

    Parameters:
        config (dict): Dictionary containing parameters for the model and evaluation process.
        dataset: The dataset to be evaluated, used for logging or selecting specific configurations.
        test_loader (DataLoader): DataLoader for the test set.
        model: The model to be evaluated.
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
    model = model
    filename = f"/mnt/data/new/modelsalex/models/{architecture_name}/svm/{config['dataset']}_{config['img_size']}.sav"
    try:
        with open(filename, 'rb') as model_file:
            ovr_classifier = pickle.load(model_file)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Model file not found: {filename}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
    #print(filename)
    #pickle.dump(ovr_classifier, open(filename, 'wb'))
    #classifier = model.get_classifier()
    #checkpoint_file = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_headonly_{architecture_name}_s{config['seed']}_best.pth"
    #checkpoint = torch.load(checkpoint_file, map_location='cpu')
    #classifier.load_state_dict(checkpoint)
    #model.head = classifier



    #model.head = classifier
    print(ovr_classifier)
    if config['task'] == "multi-label, binary-class":
        prediction = nn.Sigmoid()
    else:
        prediction = nn.Softmax(dim=1)
    # Move the model to the available device
    model = model.to(config['device'])
    model.requires_grad_(False)
    model.eval()
    y_pred, y_pred_per = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])
    embeddings_db = []
    with torch.no_grad():
         for images, labels in tqdm(test_loader):
                # Map the data to the available device
                images, labels = images.to(config['device']), labels.to(torch.float32).to(config['device'])

                output = model.forward_features(images)
                output = model.forward_head(output, pre_logits=True)
                output = output.reshape(output.shape[0], -1)
                embeddings_db.append(deepcopy(output.cpu().numpy()))
    all_outputs = np.concatenate(embeddings_db, axis=0)
    y_pred = ovr_classifier.predict(all_outputs)
    y_pred_per = ovr_classifier.predict_proba(all_outputs)


    print(f"\tFinished evaluation.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time for evaluation: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))
    return  y_pred, y_pred_per

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

    for dataset in ['breastmnist','bloodmnist','chestmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist',
                    'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist']:

        print(f"\t... for {dataset}...")
        # Extract the dataset and its metadata
        info = INFO[dataset]
        task, in_channel, num_classes = info['task'], info['n_channels'], len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        # Iterate over all image sizes
        for img_size in [28,64,128,224]:
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
                # model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Remove the classifier
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
            CORRUPTIONS_DS_KEYS = {
                'pathmnist': [
                    'pixelate', 'jpeg_compression', 'defocus_blur', 'motion_blur',
                    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
                    'saturate', 'stain_deposit', 'bubble'
                ],
                'bloodmnist': [
                    'pixelate', 'jpeg_compression', 'defocus_blur', 'motion_blur',
                    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
                    'saturate', 'stain_deposit', 'bubble'
                ],
                'dermamnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'defocus_blur', 'motion_blur',
                    'zoom_blur', 'brightness_up', 'brightness_down', 'contrast_up',
                    'contrast_down', 'black_corner', 'characters'
                ],
                'retinamnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'defocus_blur', 'motion_blur', 'brightness_down', 'contrast_down'
                ],
                'tissuemnist': [
                    'pixelate', 'jpeg_compression', 'impulse_noise', 'gaussian_blur',
                    'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down'
                ],
                'octmnist': [
                    'pixelate', 'jpeg_compression', 'speckle_noise', 'defocus_blur',
                    'motion_blur', 'contrast_down'
                ],
                'breastmnist': [
                    'pixelate', 'jpeg_compression', 'speckle_noise', 'motion_blur',
                    'brightness_up', 'brightness_down', 'contrast_down'
                ],
                'chestmnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ],
                'pneumoniamnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ],
                'organamnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ],
                'organcmnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ],
                'organsmnist': [
                    'pixelate', 'jpeg_compression', 'gaussian_noise', 'speckle_noise',
                    'impulse_noise', 'shot_noise', 'gaussian_blur', 'brightness_up',
                    'brightness_down', 'contrast_up', 'contrast_down', 'gamma_corr_up',
                    'gamma_corr_down'
                ]
            }
            config["dataset"] = dataset
            config["img_size"] = img_size
            config["architecture"] = architecture
            corruptions = CORRUPTIONS_DS_KEYS[dataset]

            for corruption in corruptions:

                # If the size of the images is 28 and its either bubbly, stain_deposit or characters the corruption doesnt exist,
                # so skip if true
                config["dataset"] = dataset
                config["img_size"] = img_size
                config["architecture"] = architecture
                if (img_size == 28 and (
                        corruption == "stain_deposit" or corruption == "bubble" or corruption == "characters")):
                    pass
                elif (img_size == 64 and (
                        corruption == "characters")):
                    pass
                elif ((img_size != 224 and (
                        corruption == "motion_blur"))):
                    pass
                else:

                    print(corruption)
                    if img_size != 224:
                        # Load the corrupted test set, according to the selected corruption
                        corrupted_test_test = CorruptedMedMNIST(
                            dataset_name=dataset,
                            corruption=corruption,
                            root=config['medmnistc_path'] + f"/{img_size}",
                            padding=total_padding,
                            as_rgb=test_dataset.as_rgb,
                            mmap_mode='r',
                            norm_mean=mean,
                            norm_std=std
                        )
                    else:
                        corrupted_test_test = CorruptedMedMNIST(
                            dataset_name=dataset,
                            corruption=corruption,
                            root=config['medmnistc_path'],
                            as_rgb=test_dataset.as_rgb,
                            padding=total_padding,
                            mmap_mode='r',
                            norm_mean=mean,
                            norm_std=std
                        )

                    test_loader = DataLoader(corrupted_test_test, batch_size=config['batch_size_eval'], shuffle=False,
                                             num_workers=4
                                             , worker_init_fn=seed_worker, generator=g)
                    y_pred, y_pred_per = evaluate(config,dataset=dataset,test_loader=test_loader, model=model)
                    y_true = test_dataset.labels
                    for severity in range(5):
                        # get probabilities of the current severity slice
                        index_range = slice(len(y_true) * severity, y_true * (severity + 1))
                        # calculate relative score and update evaluation metric
                        index_range = slice(len(y_true) * severity, len(y_true) * (severity + 1))
                        # calculate relative score and update evaluation metri
                        acc = get_ACC_kNN(y_true, y_pred[index_range], config['task'])
                        if dataset=="chestmnist":
                            y_pred_per = np.array(y_pred_per)
                            auc = get_AUC_kNN(y_true, y_pred_per[:, index_range, :], config['task'])
                        else:
                            auc = get_AUC_kNN(y_true, y_pred_per[index_range], config['task'])
                        bal_acc = get_Balanced_ACC_kNN(y_true, y_pred[index_range], config['task'])
                        co = get_Cohen_kNN(y_true, y_pred[index_range], config['task'])
                        prec = get_Precision_kNN(y_true, y_pred[index_range], config['task'])
                        d = {'dataset': dataset + corruption + str(severity), 'img_size': img_size, "Acc_Test": [acc], "Bal_Acc": [bal_acc],  "AUC": [auc],
                         "CO": [co],"Prec": [prec] }

                        dfsupport = pd.DataFrame(data=d)
                        df = pd.concat([df, dfsupport])
                        # If the size of the images is 28 and its either bubbly, stain_deposit or characters the corruption doesnt exist,
                        # so skip if true


            filename = Path(args.output_path_acc) / f"{architecture_name}/svm/corrupted/{img_size}/{dataset}_acc.csv"

            df.to_csv(filename, index=False)
    # Run the training
    #evaluate(config, train_loader, val_loader, test_loader, model)
