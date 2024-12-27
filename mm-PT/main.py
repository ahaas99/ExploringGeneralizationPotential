import time
import argparse
import yaml
import torch
import random
import numpy as np
import wandb
from model_n_data import model_n_data  # import function for data and model setup
from mulit_heat_multi_domain_pt import multi_head_multi_domain_training
from mulit_heat_multi_domain_pt_gradient_accumulation import multi_head_multi_domain_training_gradient_accumulation
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")

    start_time = time.time()
    print("\tStart main script ...")

    args = parser.parse_args()
    config_file = args.config_file

    # Load the parameters and hyperparameters of the yaml configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Set up using weights and biases from https://wandb.ai/site
    wandb.login()
    torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in cudnn
    torch.backends.cudnn.deterministic = True  # Enable cudnn to be deterministic
    torch.use_deterministic_algorithms(True)  # Enable only deterministic algorithms
    #torch.use_deterministic_algorithms(True, warn_only=True)  # Enable only deterministic algorithms
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])  # Seed the pytorch RNG for all devices (both CPU and CUDA)
    np.random.seed(config['seed'])
    g = torch.Generator()
    g.manual_seed(config['seed'])

    loader_dict = model_n_data(config=config, gen=g)
    if config["MAX_GPU_BATCH_SIZE"]==config["batch_size"]:
        multi_head_multi_domain_training(config, loader_dict)
    else:
        multi_head_multi_domain_training_gradient_accumulation(config, loader_dict)
