import torch
from tqdm import tqdm
from DEREVERB_UNET.load_config import *
from DEREVERB_UNET.utils import reconstruct_from_splits, preprocess_batches, reconstruct_from_batch
import wandb
import numpy as np
from DEREVERB_UNET.weights import save_best_weights
from DEREVERB_UNET.losses import *
config = load_config()

def tests(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"{name} is frozen")
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"{name}: running_mean={module.running_mean}, running_var={module.running_var}")





def training_loop(model, train_loader, test_loader, optimizer, scheduler, criterion, device, best_val_loss, current_config):

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    if config['Model params']['split_model']:
        # Enable gradients only for the specified heads
        for i in range((len(model.heads)+1)//2):
            if not i in current_config['current_step']['trained_heads']:
                for param_l,params_u in zip(model.heads[i].parameters(),model.heads[-i].parameters()):
                        param_l.requires_grad = False
                        params_u.requires_grad = False
                        model.heads[i].eval()
                        model.heads[-i].eval()

        # Freeze encoder and shared decoder if in "part" training phase
        if current_config['current_step']['training_phase'][1] == "part":
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.shared_decoder.parameters():
                param.requires_grad = False

            # Set frozen parts to eval mode
            model.encoder.eval()
            model.shared_decoder.eval()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

    for epoch in range(current_config['current_step']['num_epochs']):
        print(f"\nEpoch {epoch + 1} out of {current_config['current_step']['num_epochs']}")

        train_loss, model = train(model, train_loader, optimizer, criterion, device, current_config)
        val_loss, _, _ ,_= evaluate_individual_heads(model, test_loader, criterion, device, current_config)

        # Save the best weights for trained heads
        heads_idx=np.unique(current_config['current_step']['trained_heads']+(-1*current_config['current_step']['trained_heads']))
        if val_loss[heads_idx].mean() < best_val_loss[heads_idx].mean():
            best_val_loss[heads_idx] = val_loss[heads_idx]
            if config['General']['save_weights']:
                save_best_weights(model, optimizer, epoch, best_val_loss)

        if config['logging']['activate_wandb']:
            for i, loss in enumerate(val_loss):
                wandb.log({f"Head_{i}/val_loss": loss})
            wandb.log({"epoch": epoch}, commit=True)

        print(f"Epoch {epoch + 1}/{current_config['current_step']['num_epochs']}, "
              f"Train Loss: {train_loss}, "
              f"Val Loss: {val_loss} \n")

        # Scheduler step
        if current_config['current_step']['scheduler_name'] != 'CosineAnnealingLR':
            scheduler.step(val_loss[current_config['current_step']['trained_heads']].mean())
        else:
            scheduler.step()

        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")
    if config['Model params']['split_model']:
        # Reset the training states
        for i in range((len(model.heads)+1)//2):
            if not i in current_config['current_step']['trained_heads']:
                for param_l,params_u in zip(model.heads[i].parameters(),model.heads[-i].parameters()):
                        param_l.requires_grad = True
                        params_u.requires_grad = True
                        model.heads[i].train()
                        model.heads[-i].train()

        if current_config['current_step']['training_phase'][1] == "part":
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.shared_decoder.parameters():
                param.requires_grad = True
            model.encoder.train()
            model.shared_decoder.train()

    return model, best_val_loss


def train(model, train_loader, optimizer, criterion, device, current_config):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    running_losses = [0.0] * len(model.heads)

    for data, target, last_real_time_bin_per_rec_idx, last_layer_per_rec_index,actual_pixels_mask,_ in tqdm(train_loader, desc="Batch Progress", ascii=True):

        data =preprocess_batches(data,last_real_time_bin_per_rec_idx).to(device)
        target =preprocess_batches(target,last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask= preprocess_batches(actual_pixels_mask,last_real_time_bin_per_rec_idx).to(device)

        import matplotlib.pyplot as plt
        clean=target.detach().cpu().numpy()
        mask=actual_pixels_mask.detach().cpu().numpy()

        fig, axs = plt.subplots(len(clean), 1, figsize=(10, 3 * len(clean)))
        for i in range(len(clean)):
            axs[i].imshow(clean[i], aspect='auto', origin='lower')
            axs[i].set_title(f"Split Clean Chunk {i}")
        plt.tight_layout()
        plt.show()
        fig, axs = plt.subplots(len(mask), 1, figsize=(10, 3 * len(mask)))
        for i in range(len(mask)):
            axs[i].imshow(mask[i], aspect='auto', origin='lower', cmap='gray', vmin=0, vmax=1)
            axs[i].set_title(f"Mask Chunk {i}")
        plt.tight_layout()
        plt.show()


        all_outputs = model(data,actual_pixels_mask)
        optimizer.zero_grad()
        loss_w_per_sample = actual_pixels_mask.mean(dim=(1, 2))
        loss_w_per_sample /= loss_w_per_sample.sum()

        # Compute and backpropagate loss for specified heads
        for head_idx in current_config['current_step']['trained_heads']:
            if head_idx==0:
                output = all_outputs[head_idx]
                loss = (criterion[head_idx](output*actual_pixels_mask, target*actual_pixels_mask).mean(dim=(1,2))*loss_w_per_sample).mean()

            else:
                output = all_outputs[0]
                output_l, output_u = all_outputs[head_idx], all_outputs[-head_idx]
                if config['Model params']['heads_params'][head_idx]['loss'] == 'Residuals': #TODO MAKE THIS WORK WITH THE NEW WEIGHTS PER SAMPLE
                    loss_quantile_l=(criterion[head_idx](output_l*actual_pixels_mask,torch.minimum(output*actual_pixels_mask + (1 * (target*actual_pixels_mask - output*actual_pixels_mask)), output*actual_pixels_mask)).mean(axis=(1,2))*loss_w_per_sample).mean()
                    loss_quantile_u=(criterion[-head_idx](output_u*actual_pixels_mask,torch.maximum(output*actual_pixels_mask + (1 * (target*actual_pixels_mask - output*actual_pixels_mask)), output*actual_pixels_mask)).mean(axis=(1,2))*loss_w_per_sample).mean()
                else:
                    loss_quantile_l, loss_quantile_u = criterion[head_idx](output_l*actual_pixels_mask, target*actual_pixels_mask,loss_w_per_sample), criterion[-head_idx](
                        output_u*actual_pixels_mask,
                        target*actual_pixels_mask,loss_w_per_sample)
                loss=0.5*loss_quantile_u+0.5*loss_quantile_l
                if config['Model params']['heads_params'][head_idx]['loss_corr']:#TODO MAKE THIS WORK WITH THE NEW WEIGHTS PER SAMPLE
                    output = all_outputs[0]
                    interval=(output_u-output_l).reshape(output_u.shape[0],-1)
                    error=torch.abs(output-target).reshape(output_u.shape[0],-1)
                    loss_corr = spearman(interval*actual_pixels_mask,error*actual_pixels_mask,"l2",0.01)
                    loss+=0.5*loss_corr
                if config['Model params']['heads_params'][head_idx]['loss_div'] and len(current_config['current_step']['trained_heads'])>1 :#TODO MAKE THIS WORK WITH THE NEW WEIGHTS PER SAMPLE
                    loss_div=cosine_similarity_loss(head_idx,all_outputs)
                    loss +=0.1*loss_div

            loss.backward(retain_graph=True)
            running_losses[head_idx] += loss.item()
            if head_idx != 0:
                running_losses[-head_idx] += loss.item()


        optimizer.step()

    return np.array(running_losses) / len(train_loader), model


def evaluate_individual_heads(model, data_loader, criterion, device, current_config):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    running_losses = [0.0] * len(model.heads)
    all_model_outputs = []
    all_targets = []
    all_input=[]

    for data, target, last_real_time_bin_per_rec_idx, last_layer_per_rec_index, actual_pixels_mask, paths_list in tqdm(data_loader,
                                                                                                           desc="Batch Progress",
                                                                                                           ascii=True):

        data = preprocess_batches(data, last_real_time_bin_per_rec_idx).to(device)
        target = preprocess_batches(target, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = preprocess_batches(actual_pixels_mask, last_real_time_bin_per_rec_idx).to(device)
        loss_w_per_sample = actual_pixels_mask.mean(dim=(1, 2))
        loss_w_per_sample /= loss_w_per_sample.sum()

        with torch.no_grad():
            all_outputs = model(data,actual_pixels_mask)
        # Store model outputs for later use
        if current_config['current_step']['training_phase'][0]=='eval'and config['General']['save_output'][0]:
            for h_i, head in enumerate(all_outputs):
                reconstructed=reconstruct_from_batch(head.cpu().detach().squeeze().numpy(),
                                           np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0))
                for layer_i in range(reconstructed.shape[0]):
                    if paths_list[0][layer_i] in ['cal','test']:
                        folder_path_to_save=os.path.join(config['General']['save_output'][1],config['General']['save_name'],paths_list[0][layer_i],paths_list[1][layer_i])
                        os.makedirs(folder_path_to_save,exist_ok=True)
                        file_path=os.path.join(folder_path_to_save,paths_list[2][layer_i].replace('.npy',f'(layer {h_i}|{len(all_outputs)}).npy'))
                        np.save(file_path,reconstructed[layer_i])



        all_model_outputs.append(np.array([reconstruct_from_batch(head.cpu().detach().squeeze().numpy(),np.cumsum(last_real_time_bin_per_rec_idx.numpy()+1,axis=0)) for head in all_outputs]).transpose((1,0,2,3)))
        all_targets.append(reconstruct_from_batch(target.cpu().detach().squeeze().numpy(),np.cumsum(last_real_time_bin_per_rec_idx.numpy()+1,axis=0)))
        all_input.append(reconstruct_from_batch(data.cpu().detach().squeeze().numpy(),np.cumsum(last_real_time_bin_per_rec_idx.numpy()+1,axis=0)))




        # Compute losses for each head
        for output_idx in range(len(config['Model params']['heads_params'])):

            if output_idx == 0:
                output = all_outputs[output_idx]
                loss = (criterion[output_idx](output * actual_pixels_mask, target * actual_pixels_mask).mean(
                    dim=(1, 2)) * loss_w_per_sample).mean()

            else:
                output = all_outputs[0]
                output_l, output_u = all_outputs[output_idx], all_outputs[-output_idx]

                if config['Model params']['heads_params'][output_idx]['loss'] == 'Residuals': #TODO MAKE THIS WORK WITH THE NEW WEIGHTS PER SAMPLE AND MASK
                    loss_quantile_l = (criterion[output_idx](output_l,
                                                          torch.minimum(output + (1 * (target - output)), output)).mean(axis=(1,2))*loss_w_per_sample).mean()
                    loss_quantile_u = (criterion[-output_idx](output_u,
                                                           torch.maximum(output + (1 * (target - output)), output)).mean(axis=(1,2))*loss_w_per_sample).mean()

                else:

                    loss_quantile_l, loss_quantile_u = (criterion[output_idx](output_l * actual_pixels_mask,target * actual_pixels_mask,loss_w_per_sample),
                                                        criterion[-output_idx](output_u * actual_pixels_mask,target * actual_pixels_mask, loss_w_per_sample))
                loss = 0.5 * loss_quantile_u + 0.5 * loss_quantile_l
                if config['Model params']['heads_params'][output_idx]['loss_corr']: #TODO MAKE THIS WORK WITH THE NEW WEIGHTS PER SAMPLE AND MASK
                    interval = (output_u - output_l).reshape(output_u.shape[0], -1)
                    error = torch.abs(output - target).reshape(output_u.shape[0], -1)
                    loss_corr = spearman(interval, error, "l2", 0.01)
                    loss += 0.5*loss_corr
                if config['Model params']['heads_params'][output_idx]['loss_div'] and len(config['Model params']['heads_params'])>3 :#TODO MAKE THIS WORK WITH THE NEW WEIGHTS PER SAMPLE AND MASK
                    loss_div=cosine_similarity_loss(output_idx,all_outputs)
                    loss +=0.1*loss_div



            running_losses[output_idx] += loss.item()
            if output_idx != 0:
                running_losses[-output_idx] += loss.item()

    # Convert running losses to numpy array
    running_losses = 10*np.array(running_losses) / len(data_loader)
    # Concatenate outputs and targets for returning
    all_model_outputs = np.concatenate([*all_model_outputs])
    all_targets = np.concatenate([*all_targets])
    all_input = np.concatenate([*all_input])


    return running_losses, all_model_outputs, all_targets,all_input





