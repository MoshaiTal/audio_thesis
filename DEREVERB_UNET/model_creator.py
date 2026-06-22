try:
    from DEREVERB_UNET.load_config import load_config
except ModuleNotFoundError:
    from load_config import load_config


config = load_config()


def create_model():
    arch = config["Model params"]["unet_arch"]

    if arch == "stft_to_whisper_mel":
        try:
            from DEREVERB_UNET.model_corr_tree_stft_to_whisper_mel import (
                STFTToWhisperMelUNet,
            )
        except ModuleNotFoundError:
            from model_corr_tree_stft_to_whisper_mel import STFTToWhisperMelUNet

        return STFTToWhisperMelUNet(
            config["Model params"]["ngf"],
            config["General"]["mics_num"],
            config["Model params"]["kernel_size"][config["Model params"]["kernel_type"]],
            config["Model params"]["split location"],
        )

    if arch == "vanilla":
        try:
            from DEREVERB_UNET.model_corr_tree import SplitUNet
        except ModuleNotFoundError:
            from model_corr_tree import SplitUNet
        return SplitUNet(
            config["Model params"]["ngf"],
            config["General"]["mics_num"],
            config["Model params"]["kernel_size"][config["Model params"]["kernel_type"]],
            config["Model params"]["split location"],
        )

    raise ValueError(f"Unsupported unet_arch: {arch}")
