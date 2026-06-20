from DEREVERB_UNET.load_config import *
import torch
config = load_config()

def save_best_weights(model, optimizer, epoch, val_loss,model_name=""):
    save_path=config['General']['weights_path']
    file_root,suffix=os.path.splitext(save_path)
    save_path=file_root+model_name+suffix
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(state, save_path)
    print(f"Best weights saved to {save_path}")


def load_best_weights(model, load_path,val_loss=None, member_idx=None):
    """
    Loads weights into either:
    - a full model (single, non-ensemble), or
    - one member of an EnsembleUNet (if `member_idx` is provided).

    Args:
        model: full model (nn.Module or EnsembleUNet)
        load_path: checkpoint file to load from
        member_idx: if not None, loads into model.models[member_idx]
    """
    keep_DP = False
    if isinstance(model, torch.nn.DataParallel):
        gpu_ids = model.device_ids
        model = model.module
        keep_DP = True

    # checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    checkpoint = torch.load(load_path,map_location=torch.device('cpu'),weights_only=False)
    target = model.models[member_idx] if member_idx is not None else model

    target.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', -1)
    if val_loss is not None:
        val_loss[member_idx] = checkpoint.get('val_loss', None)
    else:
        val_loss = checkpoint.get('val_loss', None)
    print(f"Loaded weights from {load_path}, Epoch: {epoch}, Validation Loss: {val_loss}")

    if keep_DP:
        model = torch.nn.DataParallel(model, gpu_ids)

    return model, val_loss
