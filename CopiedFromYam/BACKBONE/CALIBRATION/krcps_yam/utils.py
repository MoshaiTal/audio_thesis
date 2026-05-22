import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

def draw_corelation(intervals,error,i):
    intervals-=intervals.min()
    intervals/=intervals.max()

    error-=error.min()
    error/=error.max()
    plt.figure()
    plt.plot(intervals,error,'.',markersize=5)
    plt.xlabel('Intervals length')
    plt.ylabel('Error length')
    plt.title(f'One sample, correlation: {spearmanr(error,intervals)[0]:.3f} , quantile loc: {i}')
    plt.show()

def calc_zero_one_loss(low,pred,up,targets):
    accuracy=np.logical_and(targets>=low,targets<=up).mean()
    prec_in = np.logical_and(pred >= low, pred <= up).mean()
    return accuracy,prec_in
def rmse(mat1,mat2):
    return [np.sqrt(np.mean((mat1) ** 2))]



def calc_correlation(mat1,mat2,func_name='pearson'):
    spearman_corrs = []
    pvs=[]
    if func_name=='spearman':
        corr_func=spearmanr
    elif func_name == 'pearson':
        corr_func = pearsonr
    else:
        raise "Correlation function not recognize"
    if mat1.ndim==1:
        return   corr_func(mat1, mat2)

    for row1, row2 in zip(mat1, mat2):
        corr, pv = corr_func(row1, row2)
        spearman_corrs.append(corr)
        pvs.append(pv)
    return np.mean(spearman_corrs) ,np.mean(pvs)

def bins_corr(mat1,mat2,k=10):
    k_hat=mat1.shape[1] % k
    matches = []
    mat1_sorted=np.argsort(mat1[:,k_hat:],axis=1).reshape(mat1.shape[0],k,-1)
    mat2_sorted=np.argsort(mat2[:,k_hat:],axis=1).reshape(mat2.shape[0],k,-1)

    for row1, row2 in zip(mat1_sorted, mat2_sorted):
        mat1_expanded = row1[..., np.newaxis]  # Shape: (n_rows, n_cols, 1)
        mat2_expanded = row2[:, np.newaxis, :]  # Shape: (n_rows, 1, n_cols)
        matches.append(np.any(mat1_expanded == mat2_expanded, axis=2).mean())

    return np.mean(matches)
def draw_spectrogram(spectrogram,title):
    plt.imshow((spectrogram.T)[::-1, :])
    plt.colorbar()
    plt.yticks([0, 64, 128, 192, 256],
               labels=[8, 6, 4, 2, 0]); plt.ylabel("Frequency [kHz]")
    plt.xticks([i/2 for i in range(0, 2*spectrogram.shape[0], 125)],
               labels=[i/250 for i in range(0, 2*spectrogram.shape[0], 125)]); plt.xlabel("Time [s]")
    plt.title(title)
    plt.show()
def load_spectrogram_data(outputs_path, targets_path):
    outputs = np.load(outputs_path)
    targets = np.load(targets_path)
    return outputs, targets

def create_spectrogram_data(outputs_path, targets_path,inputs_path=False,PLOT=False):
    filename=str(outputs_path)[str(outputs_path).find('_BIUREV_')+len('_BIUREV_'):]
    outputs, targets=load_spectrogram_data(outputs_path, targets_path)


    inputs= -1
    idx=np.random.randint(0,targets.shape[1])
    alphas=[]
    if inputs_path:
        inputs=   np.load(inputs_path)[:,0]

        # if PLOT:
        #     draw_spectrogram(inputs[idx], 'Revereb norm')



    error = np.abs(outputs[0] - targets)


    # if PLOT:
    #     draw_spectrogram(targets[idx], 'Clean')
    #     draw_spectrogram(outputs[0, idx], 'Derevereb')
    #     draw_spectrogram(error[idx] >= error[idx].mean(), 'Error')

    for i in range(1,(len(outputs)+1)//2):
        if 'error' in filename:
            mid1=outputs[0]+2*outputs[i]
            outputs[-i]=np.maximum(mid1,outputs[0])
            outputs[i]=np.minimum(mid1,outputs[0])
            # outputs[0]=outputs[0]+outputs[i]
            error = np.abs(outputs[0] - targets)

        pre_cal_intervals = outputs[-i] - outputs[i]





        accuracy_pred, precentage_in_quantile = calc_zero_one_loss(outputs[i], outputs[0], outputs[-i], targets)
        error_flat = error.reshape(error.shape[0], -1)
        interval_flat = pre_cal_intervals.reshape(pre_cal_intervals.shape[0], -1)
        pearson_corr, pv_pear = calc_correlation(error_flat, interval_flat, 'pearson')
        spearman_corr, pv_spear = calc_correlation(error_flat, interval_flat, 'spearman')
        # matches=bins_corr(error_flat, interval_flat, k=10)
        print(f"##########   Case: {filename} alpha= {np.round(accuracy_pred / 0.05) * 0.05}    #################")
        print(f"Pearson = {pearson_corr:.3f} p.v={pv_pear:.3f}, Spearman = {spearman_corr:.3f} p.v={pv_spear:.3f}")
        print(f"Accuracy in:{accuracy_pred:3f} heuristic quantile:{precentage_in_quantile:3f}")
        if PLOT:
            # draw_spectrogram(outputs[i,idx], 'Low')
            # draw_spectrogram(outputs[-i,idx], 'Up')
            # draw_spectrogram(pre_cal_intervals[idx]<=pre_cal_intervals[idx].mean(), 'Pre-cal intervals')
            draw_corelation(interval_flat[idx],error_flat[idx],i)
            sparsplot(pre_cal_intervals, error, rmse)
            sparsplot(pre_cal_intervals, error, calc_correlation)
        alphas.append(np.round(accuracy_pred / 0.05) * 0.05)

    return torch.tensor(outputs), torch.tensor(targets) ,torch.tensor(inputs),pre_cal_intervals,error,alphas


def split_to_cal_val_opt(y,cal_idx,val_idx,n_opt):
    cal, opt, val = y[cal_idx][n_opt:], y[cal_idx][:n_opt], y[val_idx]
    return cal, opt, val


def sparsplot(outputs, targets,func):
    spars_values_upper=[]
    spars_values_lower = []
    xs=np.arange(10)/10
    o=outputs.reshape(outputs.shape[0],-1)
    t=targets.reshape(targets.shape[0],-1)
    n_samaple=o.shape[0]
    n_pixels=o.shape[1]
    sorted_o_idx=np.argsort(o,axis=1)
    sorted_o = np.take_along_axis(o, sorted_o_idx, axis=1)
    sorted_t = np.take_along_axis(t, sorted_o_idx, axis=1)

    spars_values_upper.append(func(sorted_o, sorted_t)[0])
    spars_values_lower.append(func(sorted_o, sorted_t)[0])

    for x in xs[1:]:
        n_drops=int( x*n_pixels)
        res_intervals_lower = sorted_o[:,n_drops:]
        res_error_lower=sorted_t[:,n_drops:]
        spars_values_lower.append(func(res_intervals_lower,res_error_lower)[0])
        res_intervals_upper = sorted_o[:,:-n_drops]
        res_error_upper=sorted_t[:,:-n_drops]
        spars_values_upper.append(func(res_intervals_upper,res_error_upper)[0])

    plt.plot(xs,spars_values_upper,label='Upper pruning')
    plt.plot(xs,spars_values_lower,label='Lower pruning')
    plt.ylabel('Score')

    plt.xlabel('% pruning')
    plt.legend()
    plt.show()
    a=5

