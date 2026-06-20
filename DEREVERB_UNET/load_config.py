import os
import yaml


def load_config():
    config_path = os.path.join(os.getcwd(), 'DEREVERB_UNET', 'CONFIGS', r'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def load_current_config():
    config_path = os.path.join(os.path.dirname(os.getcwd()), 'DEREVERB_UNET', 'CONFIGS', r'yamnet_current_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def save_current_config(config):
    config_path = os.path.join(os.path.dirname(os.getcwd()), 'DEREVERB_UNET', 'CONFIGS', r'yamnet_current_config.yaml')
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def clear_current_config():
    config_path = os.path.join(os.path.dirname(os.getcwd()), 'DEREVERB_UNET', 'CONFIGS', r'yamnet_current_config.yaml')
    with open(config_path, 'w') as f:
        pass
