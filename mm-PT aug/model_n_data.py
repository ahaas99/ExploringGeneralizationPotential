from torch.utils.data import DataLoader, ConcatDataset
import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
import timm
from torchvision.models import alexnet, AlexNet_Weights
import torch.nn as nn
from huggingface_hub import login
from utily import seed_worker


def model_n_data(config: dict, gen):
    g = gen
    img_size = config['img_size']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    architecture_name = config['architecture']
    access_token = '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'
    # Create the model
    if architecture_name == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        # model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        # Remove the classifier
        model.classifier[6] = nn.Linear(4096, config['num_classes'])
    elif architecture_name == 'hf-hub:MahmoodLab/uni':
        login(access_token)
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5,
                                  dynamic_img_size=True)
    elif architecture_name == 'hf_hub:prov-gigapath/prov-gigapath':
        login(access_token)
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    else:
        model = timm.create_model(architecture_name, pretrained=True)
    # Image Padding with zeros, needed if img_size != 224
    if architecture_name == 'alexnet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Use ImageNet statistics
    else:
        mean, std = model.default_cfg['mean'], model.default_cfg['std']

    if architecture_name == 'hf_hub:timm/vit_base_patch14_dinov2.lvd142m':
        total_padding = max(0, 518 - img_size)
    else:
        total_padding = max(0, 224 - img_size)

    padding_left, padding_top = total_padding // 2, total_padding // 2
    padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top

    # Transformations


    mean, std = model.default_cfg['mean'], model.default_cfg['std']
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant')  # Zero-Pad the image to 224x224
    ])
    #horizontal_flip
    data_transform_aug_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant'),# Zero-Pad the image to 224x224
        transforms.RandomHorizontalFlip(p=1)
    ])
    #horizontal_flip and rotation 90 degrees
    data_transform_aug_2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant'),  # Zero-Pad the image to 224x224
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomRotation((90,90))
    ])
    #vertical_flip
    data_transform_aug_3 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant'),  # Zero-Pad the image to 224x224
        transforms.RandomVerticalFlip(p=1)
    ])

    #horizontal_flip and rotation 90 degrees
    data_transform_aug_4 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant'),  # Zero-Pad the image to 224x224
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomRotation((90,90))
    ])
    #rotation 90 degrees
    data_transform_aug_5 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant'),  # Zero-Pad the image to 224x224
        transforms.RandomRotation((90,90))
    ])
    # rotation 180 degrees
    data_transform_aug_6 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant'),  # Zero-Pad the image to 224x224
        transforms.RandomRotation((180,180))
    ])
    # rotation 270 degrees
    data_transform_aug_7 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant'),  # Zero-Pad the image to 224x224
        transforms.RandomRotation((270,270))
    ])

    # Create list of dataset names
    dataset_names = list(INFO)[:12]  # holds the names of the 12 medmnist+ datasets as a list of strings

    train_ds_dict = generate_datasets(dataset_names=dataset_names, split="train", img_size=img_size,
                                      transformations=data_transform, aug1=data_transform_aug_1, aug2=data_transform_aug_2,aug3=data_transform_aug_3,
                                      aug4=data_transform_aug_4, aug5=data_transform_aug_5, aug6=data_transform_aug_6, aug7=data_transform_aug_7)
    val_ds_dict = generate_datasets_val(dataset_names=dataset_names, split="val", img_size=img_size,
                                        transformations=data_transform)

    train_loader_dict = generate_dataloaders(dataset_dict=train_ds_dict, num_workers=num_workers,
                                             worker_seed=seed_worker, gen=g, shuffle=True, batch_size=batch_size,
                                             split="train")
    val_loader_dict = generate_dataloaders(dataset_dict=val_ds_dict, num_workers=num_workers, worker_seed=seed_worker,
                                           gen=g, shuffle=False, batch_size=batch_size, split="val")

    return {'train_loader_dict': train_loader_dict, 'val_loader_dict': val_loader_dict}


# function to generate a dictionary holding the 12 specified 2D datasets of medmnist+ instantiated with a given split
# which is train or validation
def generate_datasets(dataset_names: list[str], split: str, img_size: int, transformations, aug1=None, aug2=None,aug3=None,aug4=None,aug5=None, aug6=None, aug7=None ):
    print(f"Loading the {len(dataset_names)} datasets {dataset_names} ")
    ds_dict = {}  # dictionary that holds the 12 MEDMNIST+ dataset objects
    # fills dictionary with 12 MEDMNIST+ dataset objects
    for dataset_name in dataset_names:
        DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
        mnist_dataset = DataClass(split=split, transform=transformations, download=True, as_rgb=True,
                                  size=img_size)
        #Augment the data if breastmnist or retinamnist        
        if dataset_name == "breastmnist" or dataset_name == "retinamnist":
            mnist_dataset_aug1 = DataClass(split=split, transform=aug1, download=True, as_rgb=True,
                                           size=img_size)
            mnist_dataset_aug2 = DataClass(split=split, transform=aug2, download=True, as_rgb=True,
                                           size=img_size)
            mnist_dataset_aug3 = DataClass(split=split, transform=aug3, download=True, as_rgb=True,
                                           size=img_size)
            mnist_dataset_aug4 = DataClass(split=split, transform=aug4, download=True, as_rgb=True,
                                           size=img_size)
            mnist_dataset_aug5 = DataClass(split=split, transform=aug5, download=True, as_rgb=True,
                                           size=img_size)
            mnist_dataset_aug6 = DataClass(split=split, transform=aug6, download=True, as_rgb=True,
                                           size=img_size)
            mnist_dataset_aug7 = DataClass(split=split, transform=aug7, download=True, as_rgb=True,
                                           size=img_size)
            mnist_dataset = ConcatDataset([mnist_dataset, mnist_dataset_aug1, mnist_dataset_aug2,mnist_dataset_aug3,mnist_dataset_aug4,mnist_dataset_aug5,mnist_dataset_aug6,mnist_dataset_aug7])
        ds_dict[dataset_name] = mnist_dataset  # assign dataset as 'value' to new key with name of dataset
    return ds_dict

def generate_datasets_val(dataset_names: list[str], split: str, img_size: int, transformations):
    print(f"Loading the {len(dataset_names)} datasets {dataset_names} ")
    ds_dict = {}  # dictionary that holds the 12 MEDMNIST+ dataset objects
    # fills dictionary with 12 MEDMNIST+ dataset objects
    for dataset_name in dataset_names:
        DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
        mnist_dataset = DataClass(split=split, transform=transformations, download=True, as_rgb=True,
                                  size=img_size)

        ds_dict[dataset_name] = mnist_dataset  # assign dataset as 'value' to new key with name of dataset
    return ds_dict


# function to generate a dictionary holding the dataloaders for the 12 specified 2D datasets of medmnist+
def generate_dataloaders(dataset_dict: dict, num_workers: int, worker_seed, gen, shuffle: bool, batch_size: int,
                         split: str):
    dataloader_dict = {}  # dictionary that holds the 12 MEDMNIST+ dataset objects
    # fills dictionary with 12 MEDMNIST+ dataset objects
    for dataset_name, dataset in dataset_dict.items():
        dataloader_dict[dataset_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=num_workers, worker_init_fn=worker_seed, generator=gen)
        print(
            f"Created {split} Dataloader for {dataset_name} datasets containing {len(dataloader_dict[dataset_name])} batches")
    return dataloader_dict
