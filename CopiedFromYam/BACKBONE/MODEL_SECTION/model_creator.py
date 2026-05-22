from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import *
config = load_config()

def create_model():
    if config['Model params']['unet_arch'] == 'vanilla':
        from CopiedFromYam.BACKBONE.MODEL_SECTION.model_corr_tree import SplitUNet # from model_corr_tree import SplitUNet
        model = SplitUNet(
            config['Model params']['ngf'], 
            config['General']['mics_num'],
            config['Model params']['kernel_size'][config['Model params']['kernel_type']],
            config['Model params']['split location']
        )
    elif config['Model params']['unet_arch'] == 'multires':
        from CopiedFromYam.BACKBONE.MODEL_SECTION.error_only_model import MultiResUnet # from model_corr_tree import SplitUNet
        model = MultiResUnet(
            input_channels=config['General']['mics_num'],
            num_classes=0
        )
    elif config['Model params']['unet_arch'] == 'multires2':
        from CopiedFromYam.BACKBONE.MODEL_SECTION.model_multires import MultiResUnet # from model_corr_tree import SplitUNet
        model = MultiResUnet(
            input_channels=config['General']['mics_num'],
            num_classes=0
        )
    elif config['Model params']['unet_arch'] == 'multires all quantiles':
        from CopiedFromYam.BACKBONE.MODEL_SECTION.model_all_alphas import \
            MultiResUnet  # from model_corr_tree import SplitUNet

        model = MultiResUnet(
            input_channels=config['General']['mics_num'],
            num_classes=0
        )
    elif config['Model params']['unet_arch'] == 'ensambels':
        from CopiedFromYam.BACKBONE.MODEL_SECTION.real_ensambel_model import \
            EnsembleUNet  # from model_corr_tree import SplitUNet

        model = EnsembleUNet(
            input_channels=config['General']['mics_num'],
            num_classes=0
        )
    return model