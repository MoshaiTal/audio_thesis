from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config


config = load_config()


def create_model():
    architecture = config["Model params"]["unet_arch"]

    if architecture == "vanilla":
        from CopiedFromYam.BACKBONE.MODEL_SECTION.model_corr_tree import SplitUNet

        return SplitUNet(
            config["Model params"]["ngf"],
            config["General"]["mics_num"],
            config["Model params"]["kernel_size"][
                config["Model params"]["kernel_type"]
            ],
            config["Model params"]["split location"],
        )

    if architecture == "residual":
        from CopiedFromYam.BACKBONE.MODEL_SECTION.model_corr_tree_residual import (
            ResidualSplitUNet,
        )

        residual_cfg = config["Model params"].get("residual_output", {})
        return ResidualSplitUNet(
            ngf=config["Model params"]["ngf"],
            nc=config["General"]["mics_num"],
            kernel_size=config["Model params"]["kernel_size"][
                config["Model params"]["kernel_type"]
            ],
            split_location=config["Model params"]["split location"],
            initial_correction_scale=float(
                residual_cfg.get("initial_scale", 0.10)
            ),
            max_correction_scale=float(residual_cfg.get("max_scale", 0.50)),
            residual_channel=int(residual_cfg.get("channel", 0)),
        )

    if architecture == "multires":
        from CopiedFromYam.BACKBONE.MODEL_SECTION.error_only_model import (
            MultiResUnet,
        )

        return MultiResUnet(
            input_channels=config["General"]["mics_num"],
            num_classes=0,
        )

    if architecture == "multires2":
        from CopiedFromYam.BACKBONE.MODEL_SECTION.model_multires import (
            MultiResUnet,
        )

        return MultiResUnet(
            input_channels=config["General"]["mics_num"],
            num_classes=0,
        )

    if architecture == "multires all quantiles":
        from CopiedFromYam.BACKBONE.MODEL_SECTION.model_all_alphas import (
            MultiResUnet,
        )

        return MultiResUnet(
            input_channels=config["General"]["mics_num"],
            num_classes=0,
        )

    if architecture == "ensambels":
        from CopiedFromYam.BACKBONE.MODEL_SECTION.real_ensambel_model import (
            EnsembleUNet,
        )

        return EnsembleUNet(
            input_channels=config["General"]["mics_num"],
            num_classes=0,
        )

    raise ValueError(f"Unknown U-Net architecture: {architecture}")

