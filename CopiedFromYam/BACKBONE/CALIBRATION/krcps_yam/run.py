import os
from rcps_algo import *
from utils import *
from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.kd_algo import calibrate_k_rcps
from membership import loss_otsu
np.random.seed(55)


base_path = "/storage/yam/Thesis_Audio/dereverb_yam/outputs"
outputs_files = "test_cal_outputs_BIUREV_split1_tree_corr_quantile.npy"
targets_files = "test_cal_targets_BIUREV_split1_tree_corr_quantile.npy"
inputs_files= "test_cal_inputs_BIUREV_split1_tree_corr_quantile.npy"

outputs_path = os.path.join(base_path,outputs_files)
targets_path = os.path.join(base_path,targets_files)
inputs_path  = os.path.join(base_path,inputs_files)


hanukia_output,gt,x,rawintervals,error,alphas=create_spectrogram_data(outputs_path, targets_path,inputs_path,False)
Y=hanukia_output[[4,0,-4]]
y_pred=hanukia_output[0]
Y[1]=gt

n = Y.shape[1]
n_val = n//3
n_opt = n_val//2

split_idx = np.random.choice(n, size=n_val, replace=False).tolist()
val_idx, cal_idx=split_idx, list(set(range(n)) - set(split_idx))


cal_y,opt_y,val_y =  split_to_cal_val_opt(np.transpose(Y,(1,0,2,3)),cal_idx,val_idx,n_opt)
cal_y_pred,opt_y_pred,val_y_pred =  split_to_cal_val_opt(y_pred,cal_idx,val_idx,n_opt)
cal_y, opt_y, val_y = cal_y.permute(1, 0, 2, 3), opt_y.permute(1, 0, 2, 3), val_y.permute(1, 0, 2, 3)
cal_gt,opt_gt,val_gt =  split_to_cal_val_opt(gt,cal_idx,val_idx,n_opt)
cal_rawintervals,opt_rawintervals,val_rawintervals =  split_to_cal_val_opt(rawintervals,cal_idx,val_idx,n_opt)
cal_error,opt_error,val_error =  split_to_cal_val_opt(error,cal_idx,val_idx,n_opt)

epsilon    = 0.1
delta      = 0.1
lambda_max = torch.tensor(0.1)
stepsize   = 2e-04

_lambda,_ = rcps(cal_y, "01", "hoeffding_bentkus", epsilon, delta, lambda_max, stepsize)
_lambda_l, _lambda_u = I(_lambda,val_y[0],val_y[2])
val_intervals=_lambda_u - _lambda_l
corr,pv=calc_correlation(val_intervals.reshape(val_intervals.shape[0],-1),val_error.reshape(val_error.shape[0],-1),"spearman")
corr_pearson,pv_pearson=calc_correlation(val_intervals.reshape(val_intervals.shape[0],-1),val_error.reshape(val_error.shape[0],-1),"pearson")

venilla_corr,vanilla_pv = calc_correlation(val_rawintervals.reshape(val_rawintervals.shape[0],-1),val_error.reshape(val_error.shape[0],-1),"spearman")
venilla_corr_pearson,vanilla_pv_pearson = calc_correlation(val_rawintervals.reshape(val_rawintervals.shape[0],-1),val_error.reshape(val_error.shape[0],-1),"pearson")

rcps_mu_i = torch.mean(_lambda_u - _lambda_l)
print("####################################")
print(f"Wanted risk control: {epsilon:.4f}")
print(f"Spearman correlation before calibration: {venilla_corr:.4f}")
print(f"pearson correlation before calibration: {venilla_corr_pearson:.4f}")

print("####################################")

print(f"RCPS, mean interval length: {rcps_mu_i:.4f}")
print(f"RCPS, mean risk: {1-np.logical_and(_lambda_l<=val_y[1],_lambda_u>=val_y[1]).numpy().mean():.4f}")
print(f"RCPS spearman corr:{corr:.4f}")
print(f"RCPS pearson corr:{corr_pearson:.4f}")

print(f"Lambda: {_lambda}")
print("####################################")


k = 10
prob_size =int( n_opt*Y.shape[-1]*Y.shape[-2]*0.01)
gamma = np.linspace(0.25, 0.75, 16)

_lambda_k_temp_opt=calibrate_k_rcps(opt_y,epsilon,opt_y_pred,k,prob_size,gamma)
k, _, m = loss_otsu(cal_y[0], cal_y[2], k)
_lambda_k_mat_cal = torch.matmul(m, _lambda_k_temp_opt)

_lambda_k_cal,gain = rcps(cal_y, "01", "hoeffding_bentkus", epsilon, delta, _lambda_k_mat_cal, stepsize)

print(_lambda_k_temp_opt)
k, _, m_val = loss_otsu(val_y[0], val_y[2], k)
_lambda_k_val = torch.matmul(m_val, _lambda_k_temp_opt)+gain
_lambda_k_l_Val, _lambda_k_u_Val = I(_lambda_k_val,val_y[0],val_y[2])
val_intervals_k=_lambda_k_u_Val - _lambda_k_l_Val
corr_k,pv_k=calc_correlation(val_intervals_k.reshape(val_intervals_k.shape[0],-1),val_error.reshape(val_error.shape[0],-1),"spearman")
corr_k_person,pv_k_person=calc_correlation(val_intervals_k.reshape(val_intervals_k.shape[0],-1),val_error.reshape(val_error.shape[0],-1),"pearson")

print(f"K-RCPS, mean interval length: {val_intervals_k.mean():.4f}")
print(f"K-RCPS, mean risk: {1-np.logical_and(_lambda_k_l_Val<=val_y[1],_lambda_k_u_Val>=val_y[1]).numpy().mean():.4f}")
print(f" K-RCPS spearman corr:{corr_k:.4f}")
print(f" K-RCPS pearson corr:{corr_k_person:.4f}")

print("####################################")



