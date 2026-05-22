import numpy as np
import torch

# def preprocess_batches(tensor,indices):
#
#     """
#     Efficiently concatenate selected slices from a tensor based on indices.
#
#     Args:
#         tensor (torch.Tensor): Input tensor of shape [B, C, D1, D2, D3].
#         indices (torch.Tensor): Tensor of shape [B] with values <= C.
#
#     Returns:
#         torch.Tensor: Concatenated tensor.
#     """
#     selected_slices = [tensor[i, :indices[i]+1] for i in range(tensor.size(0))]
#     result = torch.cat(selected_slices, dim=0)
#     return result

def preprocess_batches(tensor: torch.Tensor, indices: torch.Tensor, min_total: int = 2) -> torch.Tensor:
    """
    Concatenate selected slices from each item in a batch based on indices[i],
    but guarantee that the returned batch has at least `min_total` items.

    tensor:
      - expected shape like [B, C, ...] where C is "num_slices per recording"
        (in your use: data/target/mask have a "split" dimension at axis=1)

    indices:
      - shape [B], each entry tells how many slices to keep for that recording
        (you use indices[i]+1)

    Returns:
      concatenated tensor of shape [sum_i (indices[i]+1), ...]  (>= min_total)
    """
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices)

    # Build list of per-recording slice blocks
    selected = []
    B = tensor.size(0)
    for i in range(B):
        k = int(indices[i].item()) + 1
        k = max(1, min(k, tensor.size(1)))  # clamp to valid range
        selected.append(tensor[i, :k])

    out = torch.cat(selected, dim=0)  # concat along "chunk batch" dimension

    # Guarantee at least min_total samples (avoid BatchNorm failure when N=1, H=W=1)
    if out.size(0) < min_total:
        # duplicate last sample until we reach min_total
        pad = out[-1:].repeat(min_total - out.size(0), *([1] * (out.dim() - 1)))
        out = torch.cat([out, pad], dim=0)

    return out

# def reconstruct_from_splits(splits):
#
#     if splits.ndim == 4:
#         reshaped = splits.transpose(1, 2, 0, 3).reshape(splits.shape[1], splits.shape[2], -1)
#         reshaped = reshaped[:, :80, :]
#         if reshaped.shape[-1]>=3000:
#             return reshaped[:,:,:3000]
#         pad_width = ((0, 0), (0,0), (0,3000- reshaped.shape[-1]))
#         reconstructed = np.pad(reshaped, pad_width, mode='constant', constant_values=0)  # -0.57822084
#
#     else:
#         reshaped = splits.transpose(1, 0,2).reshape(splits.shape[1], -1)
#         reshaped = reshaped[:80, :]
#         if reshaped.shape[-1]>=3000:
#             return reshaped[:,:3000]
#         pad_width = ((0, 0), (0, 3000 - reshaped.shape[-1]))
#         reconstructed = np.pad(reshaped, pad_width, mode='constant', constant_values=0)  # -0.57822084
#
#     return reconstructed
#
# def reconstruct_from_batch(batches,batch_idxs):
#     prev_s_idx=0
#     splits=[]
#     for s_idx in batch_idxs:
#         splits.append(reconstruct_from_splits(batches[prev_s_idx:s_idx]))
#         prev_s_idx = s_idx
#     return np.array(splits)


def reconstruct_from_splits(splits, mel_spec: bool = True, time_max: int = 3000, mel_bins: int = 80):
    """
    splits:
      - 3D: [Nchunks, F, Tchunk]
      - 4D: [Nchunks, M, F, Tchunk]

    Returns fixed-size arrays so we can stack across recordings:
      - mel_spec=True  -> [M, 80, time_max] or [80, time_max]
      - mel_spec=False -> [M, F,  time_max] or [F,  time_max]   (NO slicing to 80)
    """
    if splits.ndim == 4:
        # [N, M, F, Tchunk] -> [M, F, N*Tchunk]
        reshaped = splits.transpose(1, 2, 0, 3).reshape(splits.shape[1], splits.shape[2], -1)
        if mel_spec:
            reshaped = reshaped[:, :mel_bins, :]
        if reshaped.shape[-1] >= time_max:
            return reshaped[..., :time_max]
        pad_width = ((0, 0), (0, 0), (0, time_max - reshaped.shape[-1]))
        return np.pad(reshaped, pad_width, mode="constant", constant_values=0)

    # 3D: [N, F, Tchunk] -> [F, N*Tchunk]
    reshaped = splits.transpose(1, 0, 2).reshape(splits.shape[1], -1)
    if mel_spec:
        reshaped = reshaped[:mel_bins, :]
    if reshaped.shape[-1] >= time_max:
        return reshaped[:, :time_max]
    pad_width = ((0, 0), (0, time_max - reshaped.shape[-1]))
    return np.pad(reshaped, pad_width, mode="constant", constant_values=0)

def reconstruct_from_batch(batches, batch_idxs, mel_spec: bool = True, time_max: int = 3000, mel_bins: int = 80):
    prev_s_idx = 0
    out = []
    for s_idx in batch_idxs:
        out.append(reconstruct_from_splits(
            batches[prev_s_idx:s_idx],
            mel_spec=mel_spec,
            time_max=time_max,
            mel_bins=mel_bins
        ))
        prev_s_idx = s_idx
    return np.array(out)



