import os
import re

import numpy as np
from torch.utils.data import Dataset


class ReverbDataset(Dataset):
    """
    Dataset for backend-matched training:

        input  = normalized reverb_spec chunks, shape [chunks, ch, 256, 256]
        target = exact Whisper input_features from clean audio, shape [chunks, 80, 256]

    It keeps Yam's return contract, so the existing training loop can be reused.
    """

    def __init__(
        self,
        mics_num,
        dataset_name,
        data_part,
        df_or_data_base_path,
        data_base_path=None,
        data_type="spec_to_whisper_mel",
        warmup_stage=False,
    ):
        super().__init__()
        if data_base_path is None:
            data_base_path = df_or_data_base_path
        self.mics_num = mics_num
        self.data_part = data_part
        self.data_base_path = data_base_path
        self.input_type = "spec"
        self.target_type = "whisper_melspec"
        self.time_bins_res = 256
        self.target_total_T = 3000
        self.input_freq_bins = 256
        self.target_freq_bins = 80
        self.spec_pad_value = float(np.log(1e-8))

        self.input_root = os.path.join(data_base_path, "reverb_spec", data_part)
        self.target_root = os.path.join(data_base_path, "clean_whisper_melspec", data_part)
        self.clean_files, self.reverb_files = self.get_all_files_paths()
        if warmup_stage:
            self.clean_files, self.reverb_files = self.clean_files[:20], self.reverb_files[:20]

        self.norm_dir = os.path.join(data_base_path, "norm_stats")
        os.makedirs(self.norm_dir, exist_ok=True)
        self.norm_stats_path = os.path.join(self.norm_dir, "spec_train_minmax.npz")
        self.norm_min, self.norm_max = self._load_or_create_spec_train_minmax()
        self.normalized_pad_value = float(self._normalize_value(self.spec_pad_value))

        print(
            f"Dataset properties: DataSet Name: {dataset_name} Data Type: spec_to_whisper_mel\t"
            f"Data Part:{data_part} Data Base-Path:{data_base_path} Number Of Samples: {len(self.clean_files)}"
        )

    def _parse_len_from_filename(self, path):
        match = re.search(r"\[len=(\d+)\]", os.path.basename(path))
        return int(match.group(1)) if match else None

    def _valid_time_from_spec(self, arr):
        valid = np.any(np.abs(arr[: self.input_freq_bins] - self.spec_pad_value) > 1e-6, axis=0)
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            return 1
        return int(min(idx[-1] + 1, self.target_total_T))

    def _valid_time(self, clean_target_path, input_spec):
        parsed = self._parse_len_from_filename(clean_target_path)
        inferred = self._valid_time_from_spec(input_spec)
        if parsed is None:
            return inferred
        return int(max(1, min(parsed, inferred, self.target_total_T)))

    def _load_or_create_spec_train_minmax(self):
        if os.path.exists(self.norm_stats_path):
            data = np.load(self.norm_stats_path)
            return float(data["min_val"]), float(data["max_val"])

        min_val = np.inf
        max_val = -np.inf
        for root_name in ["clean_spec", "reverb_spec"]:
            root = os.path.join(self.data_base_path, root_name, "train")
            for dirpath, _, filenames in os.walk(root):
                for name in filenames:
                    if not name.endswith(".npy"):
                        continue
                    arr = np.load(os.path.join(dirpath, name)).astype(np.float32)
                    if arr.ndim != 2:
                        arr = np.squeeze(arr)
                    valid_t = self._valid_time_from_spec(arr)
                    valid = arr[: self.input_freq_bins, :valid_t]
                    if valid.size:
                        min_val = min(min_val, float(valid.min()))
                        max_val = max(max_val, float(valid.max()))
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            raise RuntimeError("Could not compute spec normalization stats.")
        if max_val <= min_val:
            max_val = min_val + 1e-6
        np.savez(self.norm_stats_path, min_val=np.float32(min_val), max_val=np.float32(max_val))
        print(f"[NORM] Created {self.norm_stats_path} with min={min_val:.6f}, max={max_val:.6f}")
        return float(min_val), float(max_val)

    def _normalize_value(self, x):
        x = np.asarray(x, dtype=np.float32)
        y = 2.0 * (x - self.norm_min) / (self.norm_max - self.norm_min) - 1.0
        return y.astype(np.float32)

    def _normalize_spec(self, arr, valid_t):
        arr = arr.astype(np.float32).copy()
        arr[: self.input_freq_bins, :valid_t] = self._normalize_value(arr[: self.input_freq_bins, :valid_t])
        if valid_t < arr.shape[-1]:
            arr[: self.input_freq_bins, valid_t:] = self.normalized_pad_value
        if arr.shape[0] > self.input_freq_bins:
            arr[self.input_freq_bins :, :] = self.normalized_pad_value
        return arr

    def split_time(self, arr, freq_bins, pad_value):
        f, t = arr.shape
        pad_f = max(0, freq_bins - f)
        pad_t = (-t) % self.time_bins_res
        arr_padded = np.pad(
            arr[:freq_bins],
            ((0, pad_f), (0, pad_t)),
            mode="constant",
            constant_values=pad_value,
        )
        return arr_padded.reshape(arr_padded.shape[0], -1, self.time_bins_res).transpose([1, 0, 2]).astype(np.float32)

    def split_channels(self, arr, pad_value):
        c, f, t = arr.shape
        pad_f = max(0, self.input_freq_bins - f)
        pad_t = (-t) % self.time_bins_res
        arr_padded = np.pad(
            arr[:, : self.input_freq_bins],
            ((0, 0), (0, pad_f), (0, pad_t)),
            mode="constant",
            constant_values=pad_value,
        )
        return arr_padded.reshape(c, arr_padded.shape[1], -1, self.time_bins_res).transpose([2, 0, 1, 3]).astype(np.float32)

    def get_all_files_paths(self):
        clean_file_path_l = []
        reverb_file_path_l = []
        for folder in os.listdir(self.target_root):
            target_subfolder = os.path.join(self.target_root, folder)
            if not os.path.isdir(target_subfolder):
                continue
            for file_name in os.listdir(target_subfolder):
                if not file_name.endswith(".npy"):
                    continue
                target_path = os.path.join(target_subfolder, file_name)
                reverb_stub = file_name[: file_name.find("[")] + file_name[file_name.find("[") :]
                reverb_path = os.path.join(self.input_root, folder, reverb_stub)
                clean_file_path_l.append(target_path)
                reverb_file_path_l.append(reverb_path)
        return clean_file_path_l, reverb_file_path_l

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, index):
        target_path, reverb_base_path = self.clean_files[index], self.reverb_files[index]

        reverb_ch_l = []
        for ch in range(1, self.mics_num + 1):
            reverb_path = (
                reverb_base_path[: reverb_base_path.find("[")]
                + f"ch{ch}_"
                + reverb_base_path[reverb_base_path.find("[") :]
            )
            reverb_ch_l.append(np.load(reverb_path).astype(np.float32))
        reverb = np.array(reverb_ch_l, dtype=np.float32)
        target = np.load(target_path).astype(np.float32)
        if target.ndim != 2:
            target = np.squeeze(target)

        actual_index = self._valid_time(target_path, reverb[0])

        for ch in range(reverb.shape[0]):
            reverb[ch] = self._normalize_spec(reverb[ch], actual_index)

        mask = np.ones((self.target_freq_bins, target.shape[-1]), dtype=np.float32)
        mask[:, actual_index:] = 0.0

        reverb = self.split_channels(reverb, self.normalized_pad_value)
        target = self.split_time(target, self.target_freq_bins, 0.0)
        mask = self.split_time(mask, self.target_freq_bins, 0.0)

        last_real_layer_idx = actual_index // self.time_bins_res
        last_real_time_bin_idx = actual_index % self.time_bins_res

        return reverb, target, last_real_layer_idx, last_real_time_bin_idx, mask, target_path.split(os.sep)[-3:]
