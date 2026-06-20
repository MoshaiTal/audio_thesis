from DEREVERB_UNET.load_config import *


config = load_config()


def create_model():
    if config["Model params"]["unet_arch"] == "vanilla_mel":
        from DEREVERB_UNET.model_corr_tree_mel import (
            MelSplitUNet,
        )

        return MelSplitUNet(
            config["Model params"]["ngf"],
            config["General"]["mics_num"],
            config["Model params"]["kernel_size"][
                config["Model params"]["kernel_type"]
            ],
            config["Model params"].get("mel split location", 3),
        )

    if config["Model params"]["unet_arch"] == "vanilla":
        from DEREVERB_UNET.model_corr_tree import SplitUNet

        return SplitUNet(
            config["Model params"]["ngf"],
            config["General"]["mics_num"],
            config["Model params"]["kernel_size"][
                config["Model params"]["kernel_type"]
            ],
            config["Model params"]["split location"],
        )

    raise ValueError(f"Unsupported U-Net architecture for this run: {config['Model params']['unet_arch']}")
