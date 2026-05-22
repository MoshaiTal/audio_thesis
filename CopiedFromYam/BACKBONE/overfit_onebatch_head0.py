import os
import copy
import numpy as np
import torch
import torch.nn as nn

from CopiedFromYam.set_env import set_device, set_seed
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.BACKBONE.MODEL_SECTION.data_loader import load_data
from CopiedFromYam.BACKBONE.MODEL_SECTION.model_creator import create_model
from CopiedFromYam.BACKBONE.MODEL_SECTION.criterions_and_optimizers_tree import set_optimizer, set_criterion
from CopiedFromYam.BACKBONE.MODEL_SECTION.utils import preprocess_batches


def grad_norm(module):
    total = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach().norm().item()
            total += g * g
            count += 1
    return (total ** 0.5), count


def main():
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config = load_config()
    device, ngpu, gpu_ids, gpu_names, multi_gpu = set_device()
    set_seed()

    # Force a simple debug setup
    current_config = {"current_step": {}}
    current_config["current_step"]["trained_heads"] = [0]
    current_config["current_step"]["training_phase"] = ["train", "full"]
    current_config["current_step"]["num_epochs"] = 1
    current_config["current_step"]["lr"] = 3e-4
    current_config["current_step"]["scheduler_name"] = "None"
    current_config["current_step"]["scheduler_params"] = {}
    current_config["current_step"]["batch_size"] = config["Train params"]["batch_size"]
    current_config["current_step"]["dropout_prob"] = config["Model params"]["dropout_prob"]
    current_config["current_step"]["optimizer_name"] = config["Train params"]["optimizer_name"]
    current_config["current_step"]["weight_decay"] = 0.0

    train_loader = load_data("train")
    first_batch = next(iter(train_loader))

    data, target, last_real_time_bin_per_rec_idx, last_layer_per_rec_index, actual_pixels_mask, meta = first_batch

    # Preprocess once, then overfit this exact processed batch
    data = preprocess_batches(data, last_real_time_bin_per_rec_idx).to(device)
    target = preprocess_batches(target, last_real_time_bin_per_rec_idx).to(device)
    actual_pixels_mask = preprocess_batches(actual_pixels_mask, last_real_time_bin_per_rec_idx).to(device)

    print("Processed batch shapes:")
    print("data:", tuple(data.shape))
    print("target:", tuple(target.shape))
    print("mask:", tuple(actual_pixels_mask.shape))

    valid_target = target[actual_pixels_mask > 0]
    valid_input = data.squeeze(1)[actual_pixels_mask > 0]

    print("VALID INPUT  min/max/mean/std:",
          valid_input.min().item(),
          valid_input.max().item(),
          valid_input.mean().item(),
          valid_input.std().item())

    print("VALID TARGET min/max/mean/std:",
          valid_target.min().item(),
          valid_target.max().item(),
          valid_target.mean().item(),
          valid_target.std().item())

    model = create_model().to(device)
    if (device.type == "cuda") and (ngpu > 1):
        model = nn.DataParallel(model, gpu_ids)

    core_model = model.module if isinstance(model, nn.DataParallel) else model

    # # Diagnostic only: remove final tanh from head 0
    # if isinstance(core_model.heads[0][-1], nn.Sequential):
    #     if isinstance(core_model.heads[0][-1][-1], nn.Tanh):
    #         core_model.heads[0][-1][-1] = nn.Identity()
    #         print("[OVERFIT-DEBUG] Replaced head 0 final Tanh with Identity")
    #     else:
    #         print("[OVERFIT-DEBUG] head 0 final layer is not Tanh:", core_model.heads[0][-1][-1])
    # else:
    #     print("[OVERFIT-DEBUG] Unexpected head 0 final block type:", type(core_model.heads[0][-1]))

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    optimizer = set_optimizer(model, current_config)
    criterion = set_criterion(current_config, device)

    # Keep only head 0 trainable for this test
    for i in range(1, (len(core_model.heads) + 1) // 2):
        for p in core_model.heads[i].parameters():
            p.requires_grad = False
        for p in core_model.heads[-i].parameters():
            p.requires_grad = False

    # Debug: confirm trainable params
    trainable = [(n, p.numel()) for n, p in core_model.named_parameters() if p.requires_grad]
    print(f"Trainable parameter groups: {len(trainable)}")
    print("First 20 trainable names:")
    for n, _ in trainable[:20]:
        print("  ", n)

    # Save initial tensors for inspection
    os.makedirs("/storage/tal/thesis/debug_overfit", exist_ok=True)
    np.save("/storage/tal/thesis/debug_overfit/input_batch.npy", data.detach().cpu().numpy())
    np.save("/storage/tal/thesis/debug_overfit/target_batch.npy", target.detach().cpu().numpy())
    np.save("/storage/tal/thesis/debug_overfit/mask_batch.npy", actual_pixels_mask.detach().cpu().numpy())

    # Optional: remember one shared decoder param to track updates
    tracked_param_name = None
    tracked_before = None
    for name, p in core_model.named_parameters():
        if "decoder" in name and p.requires_grad:
            tracked_param_name = name
            tracked_before = p.detach().clone()
            break

    print("Tracked shared decoder param:", tracked_param_name)

    steps = 400
    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        all_outputs = core_model(data, actual_pixels_mask)
        output0 = all_outputs[0]

        loss_w_per_sample = actual_pixels_mask.mean(dim=(1, 2))
        loss_w_per_sample = loss_w_per_sample / loss_w_per_sample.sum()

        raw = criterion[0](output0 * actual_pixels_mask, target * actual_pixels_mask)
        loss0 = (raw.mean(dim=(1, 2)) * loss_w_per_sample).mean()

        loss0.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                mae = torch.abs((output0 - target) * actual_pixels_mask).sum() / (actual_pixels_mask.sum() + 1e-8)

                enc_gn, enc_cnt = grad_norm(core_model.encoder)
                dec_gn, dec_cnt = grad_norm(core_model.shared_decoder)
                head0_gn, head0_cnt = grad_norm(core_model.heads[0])

                update_mean = None
                update_max = None
                if tracked_param_name is not None:
                    for name, p in core_model.named_parameters():
                        if name == tracked_param_name:
                            diff = (p.detach() - tracked_before).abs()
                            update_mean = diff.mean().item()
                            update_max = diff.max().item()
                            break

                print(
                    f"[OVERFIT] step={step:04d} "
                    f"loss0={loss0.item():.6f} "
                    f"mae={mae.item():.6f} "
                    f"enc_grad={enc_gn:.3e} "
                    f"shared_dec_grad={dec_gn:.3e} "
                    f"head0_grad={head0_gn:.3e} "
                    f"shared_dec_update_mean={update_mean if update_mean is not None else -1:.3e} "
                    f"shared_dec_update_max={update_max if update_max is not None else -1:.3e}"
                )

    # Save final prediction
    model.eval()
    with torch.no_grad():
        pred0 = core_model(data, actual_pixels_mask)[0]

    np.save("/storage/tal/thesis/debug_overfit/pred_batch_final.npy", pred0.detach().cpu().numpy())
    print("Saved debug arrays to /storage/tal/thesis/debug_overfit/")


if __name__ == "__main__":
    main()