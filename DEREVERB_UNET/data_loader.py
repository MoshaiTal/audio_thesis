import numpy as np
from torch.utils.data import DataLoader

try:
    from DEREVERB_UNET import create_dataset as dataset
    from DEREVERB_UNET.load_config import load_config
except ModuleNotFoundError:
    try:
        from DEREVERB_UNET import create_dataset as dataset
        from DEREVERB_UNET.load_config import load_config
    except ModuleNotFoundError:
        import create_dataset as dataset
        from load_config import load_config


config = load_config()


def load_data(part="train"):
    if config["General"]["dataset"] != "BIUREV":
        raise ValueError(f"Unsupported dataset: {config['General']['dataset']}")
    if part not in {"train", "val", "test", "cal"}:
        raise ValueError("Unknown dataset part, choose one of: train, val, test, cal.")

    df = config["General"]["df"] if part == "train" else 1
    ds = dataset.ReverbDataset(
        config["General"]["mics_num"],
        config["General"]["dataset"],
        part,
        df,
        config["General"]["data_base_path"],
        data_type=config["General"].get("DATA_TYPE", "melspec"),
        warmup_stage=config["General"]["warm_up_step"],
    )
    return DataLoader(
        ds,
        config["Train params"]["batch_size"],
        shuffle=(part == "train"),
        num_workers=config["Train params"]["num_workers"],
    )


def save_test_cal_raw_files(testfarloader):
    test_dataset = testfarloader.dataset
    file_paths = [str(file) for file in test_dataset.files]
    np.save("test_far_file_paths.npy", np.array(file_paths))
    print("Saved file paths for 'test_far' to test_far_file_paths.npy.")
