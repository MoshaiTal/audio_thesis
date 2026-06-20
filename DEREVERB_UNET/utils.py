import numpy as np
import torch


def preprocess_batches(tensor,indices):

    """
    Efficiently concatenate selected slices from a tensor based on indices.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, D1, D2, D3].
        indices (torch.Tensor): Tensor of shape [B] with values <= C.

    Returns:
        torch.Tensor: Concatenated tensor.
    """
    selected_slices = [tensor[i, :indices[i]+1] for i in range(tensor.size(0))]
    result = torch.cat(selected_slices, dim=0)
    return result


def reconstruct_from_splits(splits):

    if splits.ndim == 4:
        reshaped = splits.transpose(1, 2, 0, 3).reshape(splits.shape[1], splits.shape[2], -1)
        reshaped = reshaped[:, :80, :]
        if reshaped.shape[-1]>=3000:
            return reshaped[:,:,:3000]
        pad_width = ((0, 0), (0,0), (0,3000- reshaped.shape[-1]))
        reconstructed = np.pad(reshaped, pad_width, mode='constant', constant_values=0)  # -0.57822084


    else:
        reshaped = splits.transpose(1, 0,2).reshape(splits.shape[1], -1)
        reshaped = reshaped[:80, :]
        if reshaped.shape[-1]>=3000:
            return reshaped[:,:3000]
        pad_width = ((0, 0), (0, 3000 - reshaped.shape[-1]))
        reconstructed = np.pad(reshaped, pad_width, mode='constant', constant_values=0)  # -0.57822084


    return reconstructed

def reconstruct_from_batch(batches,batch_idxs):
    prev_s_idx=0
    splits=[]
    for s_idx in batch_idxs:
        splits.append(reconstruct_from_splits(batches[prev_s_idx:s_idx]))
        prev_s_idx = s_idx
    return np.array(splits)




