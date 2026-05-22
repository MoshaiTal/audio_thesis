import os
import logging
import wandb
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
import torch
import numpy as np
import random
import datetime
import pathlib
import yaml
config = load_config()


def set_device():
    # Set GPUs
    gpu_ids = config['General']['gpu_ids']
    ngpu = len(gpu_ids)
    device = torch.device("cuda:{}".format(gpu_ids[0]) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # Get GPU names
    gpu_names = []
    if torch.cuda.is_available() and ngpu > 0:
        for id in gpu_ids:
            gpu_names.append(torch.cuda.get_device_name(id))

    # Enable multi-GPU if more than one GPU is available
    multi_gpu = ngpu > 1
    print(f"Using device: {device}")
    print(f"Number of GPUs: {ngpu}")
    print(f"Selected GPU IDs: {gpu_ids}")
    print(f"Selected GPU Names: {gpu_names}")
    print(f"Multi-GPU Enabled: {multi_gpu}")

    return device, ngpu, gpu_ids, gpu_names, multi_gpu
def set_seed():
    # Set seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config['General']['seed'])
    random.seed(config['General']['seed'])
    torch.manual_seed(config['General']['seed'])

def create_check_points_file():
    # Get the current date and time
    now = datetime.datetime.now()  # datetime object containing current date and time
    dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")

    # Create directory for saving checkpoints
    train_dir = pathlib.Path('./trained_models/' + f"mics{config['General']['mics_num']}_" + dt_string)
    train_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration as a YAML file
    file_name = train_dir / 'settings.yaml'
    with open(file_name, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    print(f"Checkpoint directory created at: {train_dir}")
    print(f"Configuration saved to: {file_name}")

def logging():
    # Create log file path
    log_file = os.path.join(os.getcwd(),config['General']['checkpoints_path'],"training.log")

    # Set up logging configuration
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Add console output to logging
    logging.getLogger().addHandler(logging.StreamHandler())

    # Log seed and other configuration details
    logging.info("Random Seed: %d" % config['General']['seed'])

    # Initialize W&B if enabled
    if config['General']['use_wandb']:
        wandb.init(project=config['logging']['wandb_project'], name=config['logging']['run_name'])
        wandb.config.update(config)

    return logging

def create_weights_folder():
    # Create a folder for trained models if it doesn't exist
    if config['General']['save_weights']:
        models_dir =  pathlib.Path(os.path.join(os.getcwd(),os.path.dirname(config['General']['weights_path'])))
        models_dir.mkdir(parents=True, exist_ok=True)

def create_output_folder():
    # Create a folder for trained models if it doesn't exist
    if config['General']['save_output'][0]:
        models_dir =  pathlib.Path(os.path.join(os.getcwd(),config['General']['save_output'][1]))
        models_dir.mkdir(parents=True, exist_ok=True)
