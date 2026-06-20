import os
import re

import numpy as np
from torch.utils.data import Dataset


class ReverbDataset(Dataset):
    """
    BIUREV dataset loader for both melspec and spec.

    Compatible with both call signatures used in the project:
        ReverbDataset(mics, dataset, part, df, data_base_path, ...)
        ReverbDataset(mics, dataset, part, data_base_path, ...)

    The df argument is accepted for Yam compatibility but is not used here.
    """

    def __init__(
        self,
        mics_num,
        dataset_name,
        data_part,
        df_or_data_base_path,
        data_base_path=None,
        data_type="melspec",
        warmup_stage=False,
    ):
        super().__init__()
        if data_base_path is None:
            data_base_path = df_or_data_base_path
            self.df = 1
        else:
            self.df = df_or_data_base_path

        self.data_type = data_type
        self.mics_num = mics_num
        self.data_part = data_part
        self.data_base_path = data_base_path
        self.clean_folder_path = os.path.join(data_base_path, f"clean_{data_type}", data_part)
        self.reverb_folder_path = os.path.join(data_base_path, f"reverb_{data_type}", data_part)
        self.clean_files, self.reverb_files = self.get_all_files_paths()
        if warmup_stage:
            self.clean_files, self.reverb_files = self.clean_files[:20], self.reverb_files[:20]

        self.time_bins_res = 256
        self.target_total_T = 3000
        self.spec_pad_value = float(np.log(1e-8))
        self.final_freq_idx = 80 if self.data_type == "melspec" else 256

        self.norm_dir = os.path.join(self.data_base_path, "norm_stats")
        os.makedirs(self.norm_dir, exist_ok=True)
        self.norm_stats_path = os.path.join(self.norm_dir, f"{self.data_type}_train_minmax.npz")
        self.norm_min, self.norm_max = self._load_or_create_train_minmax()

        if self.data_type == "spec":
            self.normalized_pad_value = float(self._normalize_value(self.spec_pad_value))
        else:
            self.normalized_pad_value = float(self._normalize_value(0.0))

        print(
            f"Dataset properties: DataSet Name: {dataset_name} Data Type: {data_type}\t"
            f"Data Part:{data_part} Data Base-Path:{data_base_path} Number Of Samples: {len(self.clean_files)}"
        )

    def _parse_len_from_filename(self, path):
        match = re.search(r"\[len=(\d+)\]", os.path.basename(path))
        return int(match.group(1)) if match else None

    def _infer_valid_time_from_array(self, arr):
        if arr.ndim == 3:
            time_view = arr[:, : self.final_freq_idx, :]
            if self.data_type == "spec":
                valid = np.any(np.abs(time_view - self.spec_pad_value) > 1e-6, axis=(0, 1))
            else:
                valid = np.any(np.abs(time_view) > 1e-6, axis=(0, 1))
        else:
            time_view = arr[: self.final_freq_idx, :]
            if self.data_type == "spec":
                valid = np.any(np.abs(time_view - self.spec_pad_value) > 1e-6, axis=0)
            else:
                valid = np.any(np.abs(time_view) > 1e-6, axis=0)

        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            return 1
        return int(min(idx[-1] + 1, self.target_total_T))

    def _get_valid_time(self, clean_path, clean_arr):
        parsed = self._parse_len_from_filename(clean_path)
        inferred = self._infer_valid_time_from_array(clean_arr)
        if self.data_type == "spec":
            return inferred
        if parsed is None:
            return inferred
        return int(max(1, min(parsed, self.target_total_T)))

    def _load_or_create_train_minmax(self):
        if os.path.exists(self.norm_stats_path):
            data = np.load(self.norm_stats_path)
            return float(data["min_val"]), float(data["max_val"])

        train_clean_root = os.path.join(self.data_base_path, f"clean_{self.data_type}", "train")
        train_reverb_root = os.path.join(self.data_base_path, f"reverb_{self.data_type}", "train")

        min_val = np.inf
        max_val = -np.inf
        for folder in os.listdir(train_clean_root):
            clean_subfolder = os.path.join(train_clean_root, folder)
            reverb_subfolder = os.path.join(train_reverb_root, folder)
            if not os.path.isdir(clean_subfolder):
                continue

            for file_name in os.listdir(clean_subfolder):
                if file_name.endswith(".txt"):
                    continue
                clean_path = os.path.join(clean_subfolder, file_name)
                clean_arr = np.load(clean_path).astype(np.float32)
                valid_t = self._get_valid_time(clean_path, clean_arr)
                valid_clean = clean_arr[: self.final_freq_idx, :valid_t]
                if valid_clean.size > 0:
                    min_val = min(min_val, float(valid_clean.min()))
                    max_val = max(max_val, float(valid_clean.max()))

                for ch in range(1, self.mics_num + 1):
                    reverb_name = file_name[: file_name.find("[")] + f"ch{ch}_" + file_name[file_name.find("[") :]
                    reverb_path = os.path.join(reverb_subfolder, reverb_name)
                    if not os.path.exists(reverb_path):
                        continue
                    rev_arr = np.load(reverb_path).astype(np.float32)
                    valid_rev = rev_arr[: self.final_freq_idx, :valid_t]
                    if valid_rev.size > 0:
                        min_val = min(min_val, float(valid_rev.min()))
                        max_val = max(max_val, float(valid_rev.max()))

        if not np.isfinite(min_val) or not np.isfinite(max_val):
            raise RuntimeError(f"Could not compute normalization stats for {self.data_type}")
        if max_val <= min_val:
            max_val = min_val + 1e-6

        np.savez(self.norm_stats_path, min_val=np.float32(min_val), max_val=np.float32(max_val))
        print(f"[NORM] Created {self.norm_stats_path} with min={min_val:.6f}, max={max_val:.6f}")
        return float(min_val), float(max_val)

    def _normalize_value(self, x):
        x = np.asarray(x, dtype=np.float32)
        y = 2.0 * (x - self.norm_min) / (self.norm_max - self.norm_min) - 1.0
        return y.astype(np.float32)

    def _normalize_array(self, arr, valid_t):
        arr = arr.astype(np.float32).copy()
        arr[: self.final_freq_idx, :valid_t] = self._normalize_value(arr[: self.final_freq_idx, :valid_t])
        if valid_t < arr.shape[-1]:
            arr[: self.final_freq_idx, valid_t:] = self.normalized_pad_value
        if arr.shape[0] > self.final_freq_idx:
            arr[self.final_freq_idx :, :] = self.normalized_pad_value
        return arr.astype(np.float32)

    def split_spec(self, arr, pad_value=None):
        if pad_value is None:
            pad_value = self.normalized_pad_value if self.data_type == "spec" else 0.0

        if arr.ndim == 3:
            c, f, t = arr.shape
            pad_f = max(0, self.final_freq_idx - f)
            pad_t = (-t) % self.time_bins_res
            arr_padded = np.pad(
                arr,
                ((0, 0), (0, pad_f), (0, pad_t)),
                mode="constant",
                constant_values=pad_value,
            )
            splits = arr_padded.reshape(c, arr_padded.shape[1], -1, self.time_bins_res).transpose([2, 0, 1, 3])
        else:
            f, t = arr.shape
            pad_f = max(0, self.final_freq_idx - f)
            pad_t = (-t) % self.time_bins_res
            arr_padded = np.pad(
                arr,
                ((0, pad_f), (0, pad_t)),
                mode="constant",
                constant_values=pad_value,
            )
            splits = arr_padded.reshape(arr_padded.shape[0], -1, self.time_bins_res).transpose([1, 0, 2])
        return splits.astype(np.float32)

    def get_all_files_paths(self):
        clean_file_path_l = []
        reverb_file_path_l = []
        for folder in os.listdir(self.clean_folder_path):
            clean_subfolder_path = os.path.join(self.clean_folder_path, folder)
            if not os.path.isdir(clean_subfolder_path):
                continue
            for file_name in os.listdir(clean_subfolder_path):
                if not file_name.endswith(".txt"):
                    clean_file_path_l.append(os.path.join(clean_subfolder_path, file_name))
                    reverb_file_path_l.append(os.path.join(self.reverb_folder_path, folder, file_name))
        return clean_file_path_l, reverb_file_path_l

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        file_path_clean, file_path_reverb = self.clean_files[index], self.reverb_files[index]

        reverb_ch_l = []
        for i in range(1, self.mics_num + 1):
            file_path_reverb_new = (
                file_path_reverb[: file_path_reverb.find("[")]
                + f"ch{i}_"
                + file_path_reverb[file_path_reverb.find("[") :]
            )
            reverb_ch_l.append(np.load(file_path_reverb_new).astype(np.float32))

        reverb = np.array(reverb_ch_l, dtype=np.float32)
        clean_org = np.load(file_path_clean).astype(np.float32)
        actual_index = self._get_valid_time(file_path_clean, clean_org)

        clean_org = self._normalize_array(clean_org, actual_index)
        for ch in range(reverb.shape[0]):
            reverb[ch] = self._normalize_array(reverb[ch], actual_index)

        mask = np.ones_like(clean_org, dtype=np.float32)
        mask[self.final_freq_idx :] = 0
        mask[:, actual_index:] = 0

        mask = self.split_spec(mask, pad_value=0.0)
        reverb = self.split_spec(reverb, pad_value=self.normalized_pad_value)
        clean = self.split_spec(clean_org, pad_value=self.normalized_pad_value)

        last_real_layer_idx = actual_index // self.time_bins_res
        last_real_time_bin_idx = actual_index % self.time_bins_res

        return reverb, clean, last_real_layer_idx, last_real_time_bin_idx, mask, file_path_clean.split(os.sep)[-3:]
