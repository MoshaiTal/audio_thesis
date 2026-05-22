import data.utils as utils
import pickle
import scipy.io
import soundfile as sf
import argparse
import shutil
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from CopiedFromYam.BACKBONE.MODEL_SECTION.training_page_same_alpha import *
from set_env import *
from STT.conformer_yam import choose_interpolate_type
from CopiedFromYam.CALIBRATION.krcps_yam.calibrate_from_main import call_for_calibration_train,call_for_calibration_inf
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.set_env import set_device, set_seed
from CopiedFromYam.BACKBONE.MODEL_SECTION.model_creator import create_model
from CopiedFromYam.BACKBONE.MODEL_SECTION.data_loader import load_data
from CopiedFromYam.BACKBONE.MODEL_SECTION.criterions_and_optimizers_tree import set_criterion
from CopiedFromYam.BACKBONE.MODEL_SECTION.weights import load_checkpoint, make_ckpt_path
from CopiedFromYam.BACKBONE.MODEL_SECTION.training_page_same_alpha import evaluate_individual_heads

MICS_NUM   = 8
FRAMES_NUM = 256
K          = 512
overlap    = 0.75
eps        = 2.2204 * np.exp(-16)

# Synthesis window for the ISTFT
MAT         = scipy.io.loadmat('synt_win.mat')
synt_win    = MAT['synt_win']
SPECT_SLICE = True
# CALIB_NAME='rcps'
# SLICES_TYPE='Linear'
global _lambda, gain,FIRST_FILE
_lambda, gain = {},{}

def reconstruct_spectrogram(outputs,targets,inputs,val_loader,residue,edge_index,kk,normeliz_func,log_max_clean=None,log_min_clean=None,magnitude=None):
    pred_raw = torch.stack(outputs,dim=0).permute(1,0,2,3).squeeze().cpu()  # pred.shape == [n, 3, 256, 256] for quantile mode
    pred     = pred_raw[:,0] # main output
    clean    = targets
    reverb   = inputs[:,0] if inputs.ndim==4 else inputs
    pred_lo  = {}
    pred_hi  = {}
    for head,dic_params in config['Model params']['heads_params'].items():
        if head==0:
            continue
        if not str(dic_params['alpha']) in _lambda.keys():
            _lambda[str(dic_params['alpha'])], gain[str(dic_params['alpha'])] = call_for_calibration_train(val_loader, head,str(dic_params['alpha']),config['General']['CALIB_NAME'])
        pred_raw[:, head], pred_raw[:, -head] = call_for_calibration_inf(pred_raw[:, head], pred_raw[:, -head], pred,
                                                                         config['General']['CALIB_NAME'], _lambda[str(dic_params['alpha'])],
                                                                         gain[str(dic_params['alpha'])])

        pred_lo[str(dic_params['alpha'])] = pred_raw[:,head] # Lower quantile
        pred_hi[str(dic_params['alpha'])] = pred_raw[:,-head] # Lower quantile

    # assemble the enhanced magnitude
    recon_q_lo={}
    recon_q_hi={}
    if residue == 0:
        recon        = pred.view(edge_index * FRAMES_NUM, int(kk / 2))
        recon_reverb = reverb.view(edge_index * FRAMES_NUM, int(kk / 2))
        recon_clean  = clean.view(edge_index * FRAMES_NUM, int(kk / 2))
        for head, dic_params in config['Model params']['heads_params'].items():
            if head == 0:
                continue
            recon_q_lo[str(dic_params['alpha'])] = pred_lo[str(dic_params['alpha'])].view(edge_index * FRAMES_NUM, int(kk / 2))
            recon_q_hi[str(dic_params['alpha'])] = pred_hi[str(dic_params['alpha'])].view(edge_index * FRAMES_NUM, int(kk / 2))

    else:
        recon1 = pred[:-1, :].reshape(edge_index * FRAMES_NUM, int(kk / 2))
        recon  = torch.cat((recon1, pred[-1, -residue:, :]), axis=0)
        recon1_reverb = reverb[:-1, :].reshape(edge_index * FRAMES_NUM, int(kk / 2))
        recon_reverb  = torch.cat((recon1_reverb, reverb[-1, -residue:, :]), axis=0)
        recon1_clean  = clean[:-1, :].reshape(edge_index * FRAMES_NUM, int(kk / 2))
        recon_clean   = torch.cat((recon1_clean, clean[-1, -residue:, :]), axis=0)
        for head, dic_params in config['Model params']['heads_params'].items():
            if head == 0:
                continue

            recon1_lo = pred_lo[str(dic_params['alpha'])][:-1, :].reshape(edge_index * FRAMES_NUM, int(kk / 2))
            recon_q_lo[str(dic_params['alpha'])] = torch.cat((recon1_lo, pred_lo[str(dic_params['alpha'])][-1, -residue:, :]), axis=0)
            recon1_hi = pred_hi[str(dic_params['alpha'])][:-1, :].reshape(edge_index * FRAMES_NUM, int(kk / 2))
            recon_q_hi[str(dic_params['alpha'])] = torch.cat((recon1_hi, pred_hi[str(dic_params['alpha'])][-1, -residue:, :]), axis=0)

    if log_max_clean!=None:
        recon        = torch.cat((recon, magnitude[0, :, -1:]), axis=1)  # Add the highest frequency
        recon_clean  = torch.cat((recon_clean, magnitude[0, :, -1:]), axis=1)
        recon_reverb = torch.cat((recon_reverb, magnitude[0, :, -1:]), axis=1)
        for head, dic_params in config['Model params']['heads_params'].items():
            if head == 0:
                continue
            recon_q_lo[str(dic_params['alpha'])] = torch.cat((recon_q_lo[str(dic_params['alpha'])], magnitude[0, :, -1:]), axis=1)
            recon_q_hi[str(dic_params['alpha'])] = torch.cat((recon_q_hi[str(dic_params['alpha'])], magnitude[0, :, -1:]), axis=1)

    recon        = normeliz_func(recon, log_max_clean, log_min_clean)
    recon_reverb = normeliz_func(recon_reverb, log_max_clean, log_min_clean)
    recon_clean  = normeliz_func(recon_clean, log_max_clean, log_min_clean)
    for head, dic_params in config['Model params']['heads_params'].items():
        if head == 0:
            continue
        recon_q_lo[str(dic_params['alpha'])] = normeliz_func(recon_q_lo[str(dic_params['alpha'])], log_max_clean, log_min_clean)
        recon_q_hi[str(dic_params['alpha'])] = normeliz_func(recon_q_hi[str(dic_params['alpha'])], log_max_clean, log_min_clean)
    return recon,recon_reverb,recon_clean,recon_q_lo,recon_q_hi

def enhance_mag(magnitude, Z_cln, model, device, criterion,current_config, val_loader):
    '''
    Enhances the noisy and reverberant log-magnitude

    Parameters
    ----------
    magnitude : The log-magnitude to be enahnced, dimension are [mics_num x TIME x FREQ]
    net : The trained enhancing network
    device : Device to run on (cpu or gpu)
    '''
    # Normalise to [-1, 1]
    # max_val = np.max(magnitude)
    # min_val = np.min(magnitude)
    # magnitude = utils.normalize_log_spec(magnitude, max_val, min_val)
    K_new = K
    magnitude = utils.normalize_log_spec(magnitude, log_max_reverb, log_min_reverb)
    magnitude = torch.from_numpy(magnitude)
    magnitude = magnitude.type(torch.FloatTensor)
    Z_cln = utils.normalize_log_spec(Z_cln,log_max_reverb, log_min_reverb)
    Z_cln = torch.from_numpy(Z_cln)
    Z_cln = Z_cln.type(torch.FloatTensor)

    # Divide the input spectrogram into [256, 256] segments to match training
    edge_index = magnitude.shape[1] // FRAMES_NUM  # check how many 256-frames long segments fully fit
    residue    = magnitude.shape[1] % FRAMES_NUM
    mics_num   = magnitude.shape[0]
    # Concatenate segments along the batch dimension
    targets = Z_cln[:edge_index * FRAMES_NUM, :-1].view(-1,1, FRAMES_NUM, int(K / 2))

    if mics_num == 1:
        to_model = magnitude[:, :edge_index * FRAMES_NUM, :-1].view(-1, 1, FRAMES_NUM, int(K / 2))
        if residue != 0:
            last_part = magnitude[:, -FRAMES_NUM:, :-1].view(-1, 1, FRAMES_NUM, int(K / 2))
            to_model  = torch.cat([to_model, last_part], axis=0)
            last_part_targets = Z_cln[-FRAMES_NUM:, :-1].view(-1,1, FRAMES_NUM, int(K / 2))
            targets = torch.cat([targets, last_part_targets], axis=0)
    else:
        if residue == 0:
            to_model = torch.zeros(edge_index, mics_num, 1, FRAMES_NUM, int(K / 2))
        else:
            to_model = torch.zeros(edge_index + 1, mics_num, 1, FRAMES_NUM, int(K / 2))

        trunc_mag = magnitude[:, :edge_index * FRAMES_NUM, :-1].unsqueeze(1)
        for i in range(edge_index):
            to_model[i] = trunc_mag[:, :, i * FRAMES_NUM:(i + 1) * FRAMES_NUM, :]
        if residue != 0:
            to_model[-1] = magnitude[:, -FRAMES_NUM:, :-1].unsqueeze(1)
            last_part_targets = Z_cln[-FRAMES_NUM:, :-1].view(-1,1, FRAMES_NUM, int(K / 2))
            targets = torch.cat([targets, last_part_targets], axis=0)
        to_model = to_model.squeeze(2)
        dataset = TensorDataset(to_model, targets)
        data_loader = DataLoader(dataset, batch_size=to_model.shape[0], shuffle=False)

    _, test_cal_output, test_cal_targets, test_cal_inputs = evaluate_individual_heads(model, data_loader, criterion, device, current_config)

    test_cal_output_mel  = utils.spectrogram_to_mel(test_cal_output)
    test_cal_targets_mel = utils.spectrogram_to_mel(test_cal_targets)
    test_cal_inputs_mel  = utils.spectrogram_to_mel(test_cal_inputs[:,0]) # TODO - think of a better way than taking the first chanell
    K_mel = 160

    recon,recon_reverb,recon_clean,recon_q_lo,recon_q_hi=reconstruct_spectrogram(test_cal_output, test_cal_targets, test_cal_inputs, val_loader, residue, edge_index, K, utils.denormalize_log_spec,
                            log_max_clean, log_min_clean,magnitude)
    recon_mel, recon_reverb_mel, recon_clean_mel, recon_q_lo_mel, recon_q_hi_mel = reconstruct_spectrogram(
        test_cal_output_mel, test_cal_targets_mel, test_cal_inputs_mel, val_loader, residue, edge_index, K_mel,
        utils.normelize_mel_spec_for_whisper)

    # pred_raw = torch.stack(test_cal_output,dim=0).permute(1,0,2,3).squeeze().cpu()  # pred.shape == [n, 3, 256, 256] for quantile mode
    # pred = pred_raw[:,0] # main output
    # clean=test_cal_targets
    # reverb= test_cal_inputs[:,0] if test_cal_inputs.ndim==4 else test_cal_inputs
    # pred_lo={}
    # pred_hi={}
    # for head,dic_params in config['Model params']['heads_params'].items():
    #     if head==0:
    #         continue
    #     if not str(dic_params['alpha']) in _lambda.keys():
    #         _lambda[str(dic_params['alpha'])], gain[str(dic_params['alpha'])] = call_for_calibration_train(val_loader, head,str(dic_params['alpha']),config['General']['CALIB_NAME'])
    #     pred_raw[:, head], pred_raw[:, -head] = call_for_calibration_inf(pred_raw[:, head], pred_raw[:, -head], pred,
    #                                                                      config['General']['CALIB_NAME'], _lambda[str(dic_params['alpha'])],
    #                                                                      gain[str(dic_params['alpha'])])
    #
    #     pred_lo[str(dic_params['alpha'])]= pred_raw[:,head] # Lower quantile
    #     pred_hi[str(dic_params['alpha'])]= pred_raw[:,-head] # Lower quantile
    #
    #
    #
    # # assemble the enhanced magnitude
    # recon_q_lo={}
    # recon_q_hi={}
    # recon_q_lo_mel={}
    # recon_q_hi_mel={}
    #
    # if residue == 0:
    #     recon = pred.view(edge_index * FRAMES_NUM, int(K_new / 2))
    #     recon_reverb = reverb.view(edge_index * FRAMES_NUM, int(K_new / 2))
    #     recon_clean = clean.view(edge_index * FRAMES_NUM, int(K_new / 2))
    #     for head, dic_params in config['Model params']['heads_params'].items():
    #         if head == 0:
    #             continue
    #         recon_q_lo[str(dic_params['alpha'])] = pred_lo[str(dic_params['alpha'])].view(edge_index * FRAMES_NUM, int(K_new / 2))
    #         recon_q_hi[str(dic_params['alpha'])] = pred_hi[str(dic_params['alpha'])].view(edge_index * FRAMES_NUM, int(K_new / 2))
    #
    #
    # else:
    #     recon1 = pred[:-1, :].reshape(edge_index * FRAMES_NUM, int(K_new / 2))
    #     recon = torch.cat((recon1, pred[-1, -residue:, :]), axis=0)
    #     recon1_reverb = reverb[:-1, :].reshape(edge_index * FRAMES_NUM, int(K_new / 2))
    #     recon_reverb = torch.cat((recon1_reverb, reverb[-1, -residue:, :]), axis=0)
    #     recon1_clean = clean[:-1, :].reshape(edge_index * FRAMES_NUM, int(K_new / 2))
    #     recon_clean = torch.cat((recon1_clean, clean[-1, -residue:, :]), axis=0)
    #     for head, dic_params in config['Model params']['heads_params'].items():
    #         if head == 0:
    #             continue
    #
    #         recon1_lo = pred_lo[str(dic_params['alpha'])][:-1, :].reshape(edge_index * FRAMES_NUM, int(K_new / 2))
    #         recon_q_lo[str(dic_params['alpha'])] = torch.cat((recon1_lo, pred_lo[str(dic_params['alpha'])][-1, -residue:, :]), axis=0)
    #         recon1_hi = pred_hi[str(dic_params['alpha'])][:-1, :].reshape(edge_index * FRAMES_NUM, int(K_new / 2))
    #         recon_q_hi[str(dic_params['alpha'])] = torch.cat((recon1_hi, pred_hi[str(dic_params['alpha'])][-1, -residue:, :]), axis=0)
    #
    #
    # # recon = torch.cat((recon, magnitude[0, :, -1:]), axis=1)  # Add the highest frequency
    # # recon_clean = torch.cat((recon_clean, magnitude[0, :, -1:]), axis=1)
    # # recon_reverb = torch.cat((recon_reverb, magnitude[0, :, -1:]), axis=1)
    # # for head, dic_params in config['Model params']['heads_params'].items():
    # #     if head == 0:
    # #         continue
    # #     recon_q_lo[str(dic_params['alpha'])] = torch.cat((recon_q_lo[str(dic_params['alpha'])], magnitude[0, :, -1:]), axis=1)
    # #     recon_q_hi[str(dic_params['alpha'])] = torch.cat((recon_q_hi[str(dic_params['alpha'])], magnitude[0, :, -1:]), axis=1)
    # recon_mel=utils.normelize_mel_spec_for_whisper(recon)
    # recon_reverb_mel=utils.normelize_mel_spec_for_whisper(recon_reverb)
    # recon_clean_mel=utils.normelize_mel_spec_for_whisper(recon_clean)
    #
    # recon = utils.denormalize_log_spec(recon, log_max_clean, log_min_clean)
    # recon_reverb = utils.denormalize_log_spec(recon_reverb, log_max_clean, log_min_clean)
    # recon_clean = utils.denormalize_log_spec(recon_clean, log_max_clean, log_min_clean)
    # for head, dic_params in config['Model params']['heads_params'].items():
    #     if head == 0:
    #         continue
    #     recon_q_lo[str(dic_params['alpha'])] = utils.denormalize_log_spec(recon_q_lo[str(dic_params['alpha'])], log_max_clean, log_min_clean).numpy()
    #     recon_q_hi[str(dic_params['alpha'])] = utils.denormalize_log_spec(recon_q_hi[str(dic_params['alpha'])], log_max_clean, log_min_clean).numpy()
    #     recon_q_lo_mel[str(dic_params['alpha'])] = utils.normelize_mel_spec_for_whisper(recon_q_lo[str(dic_params['alpha'])])
    #     recon_q_hi_mel[str(dic_params['alpha'])]=utils.normelize_mel_spec_for_whisper(recon_q_hi[str(dic_params['alpha'])])

    return recon,recon_mel, recon_q_lo,recon_q_lo_mel, recon_q_hi, recon_q_hi_mel, recon_reverb,recon_reverb_mel, recon_clean, recon_clean_mel

def calc_istft_per_slice(dic_params,recon_phase,spec_hat_lo,spec_hat_hi,pred,interpolate_type='Linear',num_slices=10,do_istft=True):
    audio_slices=[]
    pred_dist=np.inf
    if type(spec_hat_lo)==dict and type(spec_hat_hi)==dict:
        for i,slice in enumerate(choose_interpolate_type(interpolate_type)(spec_hat_lo[str(dic_params['alpha'])],spec_hat_hi[str(dic_params['alpha'])], num_slices)):
            audio_slices.append(utils.istft(slice.T, recon_phase, synt_win) if do_istft else slice)
            if np.abs(slice-pred).mean() <=pred_dist:
                pred_idx  = i
                pred_dist = np.abs(slice-pred).mean()
    else:
        for i,slice in enumerate(choose_interpolate_type(interpolate_type)(spec_hat_lo,spec_hat_hi, num_slices)):
            audio_slices.append(utils.istft(slice.T, recon_phase, synt_win) if do_istft else slice)
            if np.abs(slice-pred).mean() <=pred_dist:
                pred_idx  = i
                pred_dist = np.abs(slice-pred).mean()


    return np.array(audio_slices),pred_idx


def enhance_file(reverb_path, mics_num, model, device, criterion, current_config, val_loader, div=None):
    '''
    Enhances a reverberant signal given its path

    Parameters
    ----------
    reverb_path : path to the reverbernat signal
    mics_num : number of microphones to use
    net : The trained enhancing network
    device : Device to run on (cpu or gpu)
    '''

    # Get multichannel reverb wavs
    z = []
    filename = str(reverb_path.stem)[:-4]
    folder   = filename.split("-")[1]
    z_cln, fs_c = sf.read(f"/storage/shaked/derev_clean_recs/test-clean/{folder}/{filename}.flac")
    for j in range(mics_num):
        temp, fs = sf.read(str(reverb_path)[:-6] + '{}.flac'.format(j + 1))
        z.append(temp)

    z = utils.normalize_mc(np.array(z))
    z = np.random.permutation(z)
    closest_mic = np.argmax(np.var(z, axis=-1))

    # Get the STFTs of the mutlichannel reverb signals
    temp = [utils.stft(z[m], K, overlap) for m in range(mics_num)]
    frames_num = len(temp[0][0].T)
    # print(frames_num)
    if frames_num < 256:
        raise RuntimeError('Recording is too short.')

    Z          = {}
    Z['mag']   = np.zeros((mics_num, frames_num, int(K / 2 + 1)))
    Z['phase'] = np.zeros((mics_num, frames_num, int(K / 2 + 1)))
    for i in range(mics_num):
        Z['mag'][i] = temp[i][0].T
        Z['phase'][i] = temp[i][1].T

    # Enhance the log magnitude
    Z['mag'] = np.log(Z['mag'] + eps)

    # process clean recording
    Z_cln_temp = utils.stft(z_cln, K, overlap)
    Z_cln = np.log(Z_cln_temp[0].T + eps)

    spec_hat_pred,spec_hat_pred_mel, spec_hat_lo,spec_hat_lo_mel, spec_hat_hi,spec_hat_hi_mel,spec_hat_reverb,spec_hat_reverb_mel,spec_hat_clean,spec_hat_clean_mel = enhance_mag(Z['mag'],Z_cln, model, device,criterion,current_config,val_loader)  # , save_name=f"{reverb_path.stem}")

    # Get the phase
    z = z[closest_mic]  # take only the first microphone for the phase
    z = z / 1.1 / np.max(np.abs(z))
    _, recon_phase = utils.stft(z, K, overlap)
    s_hat = utils.istft(spec_hat_pred.T, recon_phase, synt_win)
    s_hat_lo      = {}
    s_hat_hi      = {}
    s_slice       = {}
    pred_idxs     = {}
    s_slice_mel   = {}
    pred_idxs_mel = {}

    for head, dic_params in config['Model params']['heads_params'].items():
        if head == 0:
            continue
        s_hat_lo[str(dic_params['alpha'])] = utils.istft(spec_hat_lo[str(dic_params['alpha'])].T, recon_phase, synt_win)
        s_hat_hi[str(dic_params['alpha'])] = utils.istft(spec_hat_hi[str(dic_params['alpha'])].T, recon_phase, synt_win)

        s_slice[str(dic_params['alpha'])+f'_Random'],pred_idxs[str(dic_params['alpha'])+f'_Random']=calc_istft_per_slice(dic_params, recon_phase, spec_hat_lo, spec_hat_hi, spec_hat_pred, interpolate_type='Random',num_slices=10)
        s_slice[str(dic_params['alpha'])+f'_Linear'],pred_idxs[str(dic_params['alpha'])+f'_Linear']=calc_istft_per_slice(dic_params, recon_phase, spec_hat_lo, spec_hat_hi, spec_hat_pred, interpolate_type='Linear',num_slices=10)

        s_slice_mel[str(dic_params['alpha'])+f'_Random'],pred_idxs_mel[str(dic_params['alpha'])+f'_Random']=calc_istft_per_slice(dic_params, recon_phase, spec_hat_lo_mel, spec_hat_hi_mel, spec_hat_pred_mel, interpolate_type='Random',num_slices=10, do_istft=False)
        s_slice_mel[str(dic_params['alpha'])+f'_Linear'],pred_idxs_mel[str(dic_params['alpha'])+f'_Linear']=calc_istft_per_slice(dic_params, recon_phase, spec_hat_lo_mel, spec_hat_hi_mel, spec_hat_pred_mel, interpolate_type='Linear',num_slices=10, do_istft=False)


        # s_slice[str(dic_params['alpha']) + f'_Random'], pred_idxs[
        #     str(dic_params['alpha']) + f'_Random'] = calc_istft_per_slice(dic_params, recon_phase,  np.minimum(spec_hat_pred+(2*(spec_hat_clean-spec_hat_pred)),spec_hat_pred),np.maximum(spec_hat_pred+(2*(spec_hat_clean-spec_hat_pred)),spec_hat_pred), spec_hat_pred,
        #                                                                   interpolate_type='Random', num_slices=10)
        # s_slice[str(dic_params['alpha']) + f'_Linear'], pred_idxs[
        #     str(dic_params['alpha']) + f'_Linear'] = calc_istft_per_slice(dic_params, recon_phase,  np.minimum(spec_hat_pred+(2*(spec_hat_clean-spec_hat_pred)),spec_hat_pred),np.maximum(spec_hat_pred+(2*(spec_hat_clean-spec_hat_pred)),spec_hat_pred), spec_hat_pred,
        #                                                                   interpolate_type='Linear', num_slices=10)


    # Post-processing normalisation
    s_hat = s_hat / 1.1 / np.max(np.abs(s_hat))
    s_hat = s_hat[:len(z)]
    for head, dic_params in config['Model params']['heads_params'].items():
        if head == 0:
            continue
        s_hat_lo[str(dic_params['alpha'])] = s_hat_lo[str(dic_params['alpha'])] / 1.1 / np.max(np.abs(s_hat))
        s_hat_lo[str(dic_params['alpha'])] = s_hat_lo[str(dic_params['alpha'])][:len(z)]
        s_hat_hi[str(dic_params['alpha'])] = s_hat_hi[str(dic_params['alpha'])] / 1.1 / np.max(np.abs(s_hat))
        s_hat_hi[str(dic_params['alpha'])] = s_hat_hi[str(dic_params['alpha'])][:len(z)]
        s_slice[str(dic_params['alpha'])+f'_Random'] = s_slice[str(dic_params['alpha'])+f'_Random'] / 1.1 / np.max(np.abs(s_hat))
        s_slice[str(dic_params['alpha'])+f'_Random'] = s_slice[str(dic_params['alpha'])+f'_Random'][:,:len(z)]
        s_slice[str(dic_params['alpha'])+f'_Linear'] = s_slice[str(dic_params['alpha'])+f'_Linear'] / 1.1 / np.max(np.abs(s_hat))
        s_slice[str(dic_params['alpha'])+f'_Linear'] = s_slice[str(dic_params['alpha'])+f'_Linear'][:,:len(z)]
    return z, s_hat, z_cln, fs_c, s_hat_lo, s_hat_hi,spec_hat_pred,spec_hat_pred_mel, spec_hat_lo,spec_hat_lo_mel, spec_hat_hi,spec_hat_hi_mel,spec_hat_reverb,spec_hat_reverb_mel,spec_hat_clean,spec_hat_clean_mel,s_slice,s_slice_mel, pred_idxs, pred_idxs_mel


def enhance_scenario(wavs_dir, results_audio_dir,results_spectograms_dir,results_mel_spectograms_dir, scenario_file, mics_num, model, device,criterion,current_config,config,val_loader):
    """
    Creates spectrograms of size 256x257 from the validation/test data

    Parameters
    ----------
    wavs_dir: directory with multichannel reverberant speech.
    results_dir : directory to save the enhanced files.
    scenario_file: text files listing all the reverberant files.
    mics_num: number of used microphones.
    net : trained pytorch model for dereverberation.
    device: cuda/cpu.
    trained_model_path: the path to the trained model file.
    """
    fs = 16000

    # Extract the reverbernat WAV files names
    with open(scenario_file, 'r') as f:
        reverb_files = f.readlines()
    reverb_files = [x.strip() for x in reverb_files]
    trial_name = config['General']['save_name']

    # save_dir = results_audio_dir / scenario_file.name[:7]
    save_dir_audio = pathlib.Path(results_audio_dir) / trial_name
# / scenario_file.name[:7]
    save_dir_audio.mkdir(parents=True, exist_ok=True)
    save_dir_spectograms = pathlib.Path(results_spectograms_dir) / trial_name # / scenario_file.name[:7]
    save_dir_spectograms.mkdir(parents=True, exist_ok=True)

    save_dir_mel_spectograms = pathlib.Path(results_mel_spectograms_dir) / trial_name # / scenario_file.name[:7]
    save_dir_mel_spectograms.mkdir(parents=True, exist_ok=True)

    # processed = 1
    total = 1
    skipped = 0
    skipped_files = []
    for i, file in enumerate(reverb_files):
        reverb_path = wavs_dir / file[1:]
        div = file[1:].split("/")[1]
        print(f'processing {reverb_path} ({total}/{len(reverb_files)}) | skipped {skipped}')
        total += 1
        z, s_hat, s_clean, fs_clean, s_hat_lo, s_hat_hi, spec_hat_pred,spec_hat_pred_mel, spec_hat_lo,spec_hat_lo_mel, spec_hat_hi,spec_hat_hi_mel,spec_hat_reverb,spec_hat_reverb_mel,spec_hat_clean,spec_hat_clean_mel, s_slice, s_slice_mel, pred_idxs ,pred_idxs_mel = enhance_file(
            reverb_path, mics_num, model, device, criterion, current_config, val_loader, div)
        # try:
        #         z, s_hat, s_clean,fs_clean, s_hat_lo, s_hat_hi,spec_hat_pred,spec_hat_lo, spec_hat_hi,spec_hat_reverb,spec_hat_clean,s_slice,pred_idxs = enhance_file(reverb_path, mics_num, model, device,criterion,current_config,val_loader,div)
        # except:  # In case the signal is too short
        #     skipped += 1
        #     skipped_files.append(reverb_path.name)
        #     continue

        enhanced_save_name = save_dir_audio / file[1:]
        enhanced_save_name.parent.mkdir(parents=True, exist_ok=True)

        enhanced_save_name_spectograms = save_dir_spectograms / file[1:]
        enhanced_save_name_spectograms.parent.mkdir(parents=True, exist_ok=True)

        enhanced_save_name_mel_spectograms = save_dir_mel_spectograms / file[1:]
        enhanced_save_name_mel_spectograms.parent.mkdir(parents=True, exist_ok=True)

        sf.write(enhanced_save_name, s_hat, fs)
        clean_name = save_dir_audio / (file[1:-5] + "_clean.flac")
        sf.write(clean_name, s_clean, fs_clean)
        reverb_name=save_dir_audio / (file[1:-5] + "_reverb.flac")
        shutil.copy(reverb_path,reverb_name)
        # saving quantile results as sound files

        for head, dic_params in config['Model params']['heads_params'].items():
            if head == 0:
                continue
            if 'error' in trial_name.lower():
                interval_method = 'error'
            else:
                interval_method = f"alpha={str(dic_params['alpha'])}"
            quantile_lo_name = save_dir_audio / (file[1:-5] + f"_lower_{interval_method}.flac")
            quantile_hi_name = save_dir_audio / (file[1:-5] + f"_upper_{interval_method}.flac")
            sf.write(quantile_lo_name, s_hat_lo[str(dic_params['alpha'])], fs) # s_hat_lo[str(dic_params['alpha'])] is a (1000,) ndarray
            sf.write(quantile_hi_name, s_hat_hi[str(dic_params['alpha'])], fs)

            pred_index_random_name = save_dir_audio / (file[1:-5] + f"_pred_index_{interval_method}_Random.npy")
            pred_index_linear_name = save_dir_audio / (file[1:-5] + f"_pred_index_{interval_method}_Linear.npy")
            np.save(pred_index_random_name,np.array(pred_idxs[str(dic_params['alpha']) + f'_Random']))
            np.save(pred_index_linear_name,np.array(pred_idxs[str(dic_params['alpha']) + f'_Linear']))

            pred_index_random_name_mel = save_dir_mel_spectograms / (file[1:-5] + f"_pred_index_{interval_method}_Random.npy")
            pred_index_linear_name_mel = save_dir_mel_spectograms / (file[1:-5] + f"_pred_index_{interval_method}_Linear.npy")
            np.save(pred_index_random_name_mel,np.array(pred_idxs_mel[str(dic_params['alpha']) + f'_Random']))
            np.save(pred_index_linear_name_mel,np.array(pred_idxs_mel[str(dic_params['alpha']) + f'_Linear']))

            quantile_lo_name_spectograms = save_dir_spectograms / (file[1:-5] + f"_lower_{interval_method}")
            quantile_hi_name_spectograms = save_dir_spectograms / (file[1:-5] + f"_upper_{interval_method}")
            np.save(quantile_lo_name_spectograms, spec_hat_lo[str(dic_params['alpha'])])
            np.save(quantile_hi_name_spectograms, spec_hat_hi[str(dic_params['alpha'])])

            # quantile_lo_name_mel_spectograms = save_dir_mel_spectograms / (file[1:-5] + f"_lower_{interval_method}")
            # quantile_hi_name_mel_spectograms = save_dir_mel_spectograms / (file[1:-5] + f"_upper_{interval_method}")
            # np.save(quantile_lo_name_mel_spectograms, spec_hat_lo_mel[str(dic_params['alpha'])])
            # np.save(quantile_hi_name_mel_spectograms, spec_hat_hi_mel[str(dic_params['alpha'])])

            slices_name_audios_random =[ save_dir_audio / (file[1:-5] + f"_{interval_method}_slices_{s}_Random.flac") for s in range(s_slice[str(dic_params['alpha'])+f'_Random'].shape[0]) ]
            [sf.write(p, s_slice[str(dic_params['alpha'])+f'_Random'][ii,:], fs) for ii,p in enumerate(slices_name_audios_random)]  #  s_slice[str(dic_params['alpha'])] is a (100,10000) ndarray

            slices_name_audios_linear = [save_dir_audio / (file[1:-5] + f"_{interval_method}_slices_{s}_Linear.flac") for s in
                                    range(s_slice[str(dic_params['alpha']) + f'_Linear'].shape[0])]
            [sf.write(p, s_slice[str(dic_params['alpha']) + f'_Linear'][ii, :], fs) for ii, p in
             enumerate(slices_name_audios_linear)]

            slices_name_spectrograms_mel_random = [save_dir_mel_spectograms / (file[1:-5] + f"_{interval_method}_slices_{s}_Random.npy")
                                         for s in range(s_slice_mel[str(dic_params['alpha']) + f'_Random'].shape[0])]
            [np.save(p, s_slice_mel[str(dic_params['alpha']) + f'_Random'][ii, :]) for ii, p in
             enumerate(slices_name_spectrograms_mel_random)]  # s_slice[str(dic_params['alpha'])] is a (100,10000) ndarray

            slices_name_spectrograms_mel_linear = [save_dir_mel_spectograms / (file[1:-5] + f"_{interval_method}_slices_{s}_Linear.npy")
                                         for s in
                                         range(s_slice_mel[str(dic_params['alpha']) + f'_Linear'].shape[0])]
            [np.save(p, s_slice_mel[str(dic_params['alpha']) + f'_Linear'][ii, :]) for ii, p in
             enumerate(slices_name_spectrograms_mel_linear)]

        np.save(enhanced_save_name_spectograms,spec_hat_pred)
        clean_name_spectograms = save_dir_spectograms / (file[1:-5] + "_clean")
        np.save(clean_name_spectograms,spec_hat_clean)

        np.save(enhanced_save_name_mel_spectograms,spec_hat_pred_mel)
        clean_name_mel_spectograms = save_dir_mel_spectograms / (file[1:-5] + "_clean")
        np.save(clean_name_mel_spectograms,spec_hat_clean_mel)

        reverb_name_spectograms = save_dir_spectograms/ (file[1:-5] + "_reverb")
        np.save(reverb_name_spectograms,spec_hat_reverb)

        reverb_name_mel_spectograms = save_dir_mel_spectograms/ (file[1:-5] + "_reverb")
        np.save(reverb_name_mel_spectograms,spec_hat_reverb_mel)
        # saving quantile results as sound files
        a = 5

    print(f"Skipped {skipped} files ({len(skipped_files)} files in skip list)")
    with open("skipped_files.txt", "w") as skips:
        for fl in skipped_files:
            skips.write(fl + "\n")

def main(model, criterion, device,current_config, config, cal_loader):
    if config['General']['CALIB_NAME'] not in ['rcps','krcps']:
        config['General']['CALIB_NAME']=''

    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', help='BIUREV/BIUREV-N', type=str, default='BIUREV', choices=['BIUREV', 'BIUREV-N'])
    args = parser.parse_args()
    min_max_file = f'./data/spectrograms/BIUREV/mics{2}/train/global_min_max.p'

    with open(min_max_file, 'rb') as f:
        global log_max_clean, log_min_clean, log_max_reverb, log_min_reverb
        log_max_clean, log_min_clean, log_max_reverb, log_min_reverb = pickle.load(f)
    # ' for uncalibrated

    wavs_dir                    = pathlib.Path(f"/storage/shaked/derev_recs/{args.dataset}")
    results_audio_dir           = pathlib.Path(f"/storage/tal/thesis/CopiedFromYam/STT/data/Audio_trials_{ config['General']['CALIB_NAME']}")
    results_spectograms_dir     = pathlib.Path(f"/storage/tal/thesis/CopiedFromYam/STT/data/Spectrogram_trials_{ config['General']['CALIB_NAME']}")
    results_mel_spectograms_dir = pathlib.Path(f"/storage/tal/thesis/CopiedFromYam/STT/data/Mel_Spectrogram_trials_{ config['General']['CALIB_NAME']}")

    dists = ["far"]
    for dist in dists:
        scenario_file = pathlib.Path('./taskfiles_new/SimData_et_for_' + dist)
        enhance_scenario(wavs_dir, results_audio_dir,results_spectograms_dir,results_mel_spectograms_dir, scenario_file, 2, model, device,criterion,current_config,config,cal_loader)


def resolve_best_ckpt(cfg: dict) -> str:
    base = cfg["General"]["weights_path"]  # /storage/tal/thesis/weights/ensambels3.pth
    best_path = make_ckpt_path(base, "best")  # -> ensambels3_best.pth
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(base):
        print(f"WARNING: best checkpoint not found, falling back to base path: {base}")
        return base
    raise FileNotFoundError(f"Could not find checkpoint. Tried: {best_path} and {base}")


if __name__ == "__main__":
    config = load_config()
    set_seed()
    device, ngpu, gpu_ids, gpu_names, multi_gpu = set_device()
    print(f"Using device: {device}, ngpu={ngpu}, multi_gpu={multi_gpu}")

    # Build model
    model = create_model().to(device)
    if (device.type == "cuda") and (ngpu > 1):
        model = nn.DataParallel(model, gpu_ids)

    # Load BEST checkpoint
    best_ckpt = resolve_best_ckpt(config)
    model, _, start_epoch, best_val_loss = load_checkpoint(model=model, optimizer=None, load_path=best_ckpt, map_location="cpu")
    print(f"Loaded BEST checkpoint: {best_ckpt} (ckpt epoch={start_epoch-1})")

    # Force eval mode behavior in evaluation function
    current_config = {"current_step": {}}
    current_config["current_step"]["training_phase"] = ["eval", "whole"]
    current_config["current_step"]["trained_heads"] = ["all"]

    # criterion list (used inside evaluate_individual_heads)
    criterion = set_criterion(current_config, device)

    # -----------------------------
    # Build CALIBRATION TUPLE for RCPS/KRCPS:
    # val_tuple = (val_output, val_target, None)
    # where val_output is torch of shape [num_heads, N, H, W]
    # -----------------------------
    cal_loader = load_data("cal")  # your DataBase calibration split

    cal_loss, cal_outputs_np, cal_targets_np, _ = evaluate_individual_heads(model, cal_loader, criterion, device, current_config)

    # cal_outputs_np: [N, num_heads, H, W]  -> torch: [num_heads, N, H, W]
    cal_outputs = torch.tensor(cal_outputs_np, dtype=torch.float32).permute(1, 0, 2, 3)
    cal_targets = torch.tensor(cal_targets_np, dtype=torch.float32)  # [N, H, W]

    val_tuple = (cal_outputs, cal_targets, None)

    print(f"Calibration tuple shapes: outputs={cal_outputs.shape}, targets={cal_targets.shape}")

    # -----------------------------
    # Run BIUREV export/eval:
    # Your export script's main expects: main(model, criterion, device, current_config, config, cal_loader)
    # BUT 'cal_loader' must be the tuple for RCPS/KRCPS
    # -----------------------------
    main(model, criterion, device, current_config, config, val_tuple)