import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def freeze_backbone(model, METHOD):
    for param in model.parameters():
        param.requires_grad = False
    if METHOD != 'condformer':
        for param in model.model.encoder.noise_to_gamma.parameters():
            param.requires_grad = True
        for param in model.model.encoder.noise_to_beta.parameters():
            param.requires_grad = True
    else:
        for param in model.model.encoder.condblock.parameters():
            param.requires_grad = True


def _basename(path_or_name: str) -> str:
    return os.path.basename(path_or_name)


def canonical_sample_stem(path_or_name: str, wanted_ch: str = "ch1") -> Optional[str]:
    """Return a canonical utterance stem shared by clean/reverb/pred/lower/upper/text files.

    Examples handled:
      clean:  cal_00000_[len=1662].npy                -> cal_00000_[len=1662]
      reverb: cal_00000_ch1_[len=1662].npy            -> cal_00000_[len=1662]
      pred:   cal_00000_[len=1662]_ch1_pred.npy       -> cal_00000_[len=1662]
      lower:  cal_00000_[len=1662]_ch1_lower_alpha=0.2.npy -> cal_00000_[len=1662]
      upper:  cal_00000_[len=1662]_ch1_upper_alpha=0.2.npy -> cal_00000_[len=1662]
      text:   cal_00000_[len=1662].txt                -> cal_00000_[len=1662]
    """
    name = Path(path_or_name).stem

    # New dereverb naming from generate_audiofiles_tal.py
    pat = rf"^(?P<stem>.+?)_{re.escape(wanted_ch)}_(pred|lower_alpha=[^_]+|upper_alpha=[^_]+)$"
    m = re.match(pat, name)
    if m:
        return m.group("stem")

    # Reverb naming: insert channel before [len=...]
    m = re.match(rf"^(?P<prefix>.+?)_{re.escape(wanted_ch)}(?P<suffix>_\[len=.*\])$", name)
    if m:
        return f"{m.group('prefix')}{m.group('suffix')}"

    # Clean/text or other already-canonical names
    return name



def detect_file_role(path_or_name: str, wanted_ch: str = "ch1", wanted_alpha: Optional[str] = None):
    path_str = str(path_or_name)
    name = _basename(path_str)

    if name.endswith('.txt'):
        return 'text'

    if 'clean_melspec' in path_str:
        return 'clean'

    if 'reverb_melspec' in path_str and wanted_ch in name:
        return 'reverb'

    if f"_{wanted_ch}_pred" in name:
        return 'pred'

    if f"_{wanted_ch}_lower_alpha=" in name:
        if wanted_alpha is None or f"lower_alpha={wanted_alpha}" in name:
            return 'lower'
        return False

    if f"_{wanted_ch}_upper_alpha=" in name:
        if wanted_alpha is None or f"upper_alpha={wanted_alpha}" in name:
            return 'upper'
        return False

    # Backward compatibility with old layer-based naming
    match = re.search(r'\(layer (\d+)\|(\d+)\)', name)
    if match:
        x = int(match.group(1))
        if x == 1:
            return 'lower'
        if x == 2:
            return 'upper'
        if x == 0:
            return 'pred'

    return False



def group_files_by_name(folder_path, groups=False, WANTED_CH="ch1", WANTED_ALPHA=None):
    """Backward-compatible grouping, but understands current pred/lower/upper naming."""
    if not groups:
        groups = defaultdict(list)
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        key = detect_file_role(full_path, WANTED_CH, WANTED_ALPHA)
        if key:
            groups[key].append(full_path)
    for group in groups:
        groups[group].sort(key=lambda p: canonical_sample_stem(p, WANTED_CH) or _basename(p))
    return groups



def remove_unwanted_text_path(groups_text, groups_pred, WANTED_CH="ch1"):
    pred_stems = {canonical_sample_stem(pred, WANTED_CH) for pred in groups_pred}
    filtered_files = [
        file for file in groups_text
        if canonical_sample_stem(file, WANTED_CH) in pred_stems
    ]
    return filtered_files



def build_paired_file_rows(
    output_folder_path: str,
    clean_folder_path: str,
    reverb_folder_path: str,
    targets_folder_path: str,
    WANTED_CH: str = "ch1",
    WANTED_ALPHA: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return paired rows by canonical utterance stem.

    Each returned row contains keys among: pred, lower, upper, clean, reverb, text, stem.
    Only fully matched rows are returned.
    """
    by_stem: Dict[str, Dict[str, str]] = defaultdict(dict)

    def ingest(folder: str, kind_source: str):
        if not os.path.isdir(folder):
            return
        for file_name in os.listdir(folder):
            full_path = os.path.join(folder, file_name)
            role = detect_file_role(full_path, WANTED_CH, WANTED_ALPHA)
            if not role:
                continue
            stem = canonical_sample_stem(full_path, WANTED_CH)
            if stem is None:
                continue
            by_stem[stem][role] = full_path
            by_stem[stem]['stem'] = stem

    ingest(output_folder_path, 'output')
    ingest(clean_folder_path, 'clean')
    ingest(reverb_folder_path, 'reverb')
    ingest(targets_folder_path, 'text')

    rows = []
    required = ['pred', 'lower', 'upper', 'clean', 'reverb', 'text']
    for stem in sorted(by_stem.keys()):
        row = by_stem[stem]
        if all(k in row for k in required):
            rows.append(row)
    return rows
