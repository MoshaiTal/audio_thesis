import numpy as np
import inspect
from torch.utils.data import DataLoader

try:
    from DEREVERB_UNET import create_dataset as dataset
    from DEREVERB_UNET.load_config import load_config
except ModuleNotFoundError:
    try:
        from DEREVERB_UNET import create_dataset as dataset
        from DEREVERB_UNET.load_config import load_config
    except ModuleNotFoundError:
        from DEREVERB_UNET import create_dataset as dataset
        from DEREVERB_UNET.load_config import load_config


config = load_config()


def _data_type():
    return config["General"].get("DATA_TYPE", "melspec")


def load_data(part="train"):
    if config["General"]["dataset"] != "BIUREV":
        raise ValueError(f"Unsupported dataset: {config['General']['dataset']}")

    if part not in {"train", "val", "test", "cal"}:
        raise ValueError("Unknown dataset part, choose one of: train, val, test, cal.")

    shuffle = part == "train"
    drop_last = part == "train"
    df = config["General"]["df"] if part == "train" else 1
    dataset_kwargs = {
        "data_type": _data_type(),
        "warmup_stage": config["General"]["warm_up_step"],
    }
    dataset_args = [
        config["General"]["mics_num"],
        config["General"]["dataset"],
        part,
    ]

    # Two project copies exist:
    #   old Yam: ReverbDataset(mics, dataset, part, df, data_base_path, ...)
    #   newer:   ReverbDataset(mics, dataset, part, data_base_path, ...)
    params = list(inspect.signature(dataset.ReverbDataset.__init__).parameters)
    if "df" in params:
        dataset_args.extend([df, config["General"]["data_base_path"]])
    else:
        dataset_args.append(config["General"]["data_base_path"])

    return DataLoader(
        dataset.ReverbDataset(*dataset_args, **dataset_kwargs),
        config["Train params"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["Train params"]["num_workers"],
        drop_last=drop_last,
    )


def save_test_cal_raw_files(testfarloader):
    test_dataset = testfarloader.dataset
    file_paths = [str(file) for file in test_dataset.files]
    np.save("test_far_file_paths.npy", np.array(file_paths))
    print("Saved file paths for 'test_far' to test_far_file_paths.npy.")
