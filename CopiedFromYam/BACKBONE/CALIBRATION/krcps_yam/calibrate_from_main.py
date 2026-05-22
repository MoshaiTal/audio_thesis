from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.rcps_algo import *
from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.utils import *
from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.kd_algo import calibrate_k_rcps
from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.membership import loss_otsu

np.random.seed(55)

def run_calibration(config):
    _lambda, gain = {}, {}

    save_dir = config['General']['save_output'][1]
    dereverb_test=torch.tensor(np.load(f"{save_dir}/test_outputs_{config['General']['dataset']}_{config['General']['save_name']}.npy")).permute(1, 0, 2, 3)
    dereverb_cal=torch.tensor(np.load(f"{save_dir}/cal_outputs_{config['General']['dataset']}_{config['General']['save_name']}.npy")).permute(1, 0, 2, 3)
    clean_cal=torch.tensor(np.load(f"{save_dir}/cal_targets_{config['General']['dataset']}_{config['General']['save_name']}.npy"))
    pred_lo = {}
    pred_hi = {}
    for head, dic_params in config['Model params']['heads_params'].items():
        if head == 0:
            continue
        if not str(dic_params['alpha']) in _lambda.keys():
            _lambda[str(dic_params['alpha'])], gain[str(dic_params['alpha'])] = call_for_calibration_train((dereverb_cal,clean_cal,None), head,
                                                                                                           str(dic_params['alpha']), config['General']['CALIB_NAME'])

        dereverb_test[head], dereverb_test[-head] = call_for_calibration_inf(dereverb_test[head], dereverb_test[-head], dereverb_test[0],
                                                                         config['General']['CALIB_NAME'],
                                                                         _lambda[str(dic_params['alpha'])],
                                                                         gain[str(dic_params['alpha'])])

        pred_lo[str(dic_params['alpha'])] = dereverb_test[head]  # Lower quantile
        pred_hi[str(dic_params['alpha'])] = dereverb_test[-head]  # Lower quantile
    a=5


def kd_calibrate_train(n_opt, opt_y_pred, cal_y, opt_y, epsilon ,delta = 0.1,stepsize = 2e-04, k = 10,prob=0.01):

    prob_size =int( n_opt*cal_y.shape[-1]*cal_y.shape[-2]*prob)
    gamma = np.linspace(0.25, 0.75, 16)
    _lambda_k_temp_opt=calibrate_k_rcps(opt_y,epsilon,opt_y_pred,k,prob_size,gamma)
    k, _, m = loss_otsu(cal_y[0], cal_y[2], k)
    _lambda_k_mat_cal = torch.matmul(m, _lambda_k_temp_opt)
    _lambda_k_cal,gain = rcps(cal_y, "01", "hoeffding_bentkus", epsilon, delta, _lambda_k_mat_cal, stepsize)
    return    _lambda_k_temp_opt,gain

def call_for_calibration_train(val_loader,head_idx,alpha,calib_name):
    if not calib_name in ['rcps','krcps']:
        return False,False
    gain=None
    val_output, val_target, _=val_loader
    a=5
    Y = torch.stack([val_output[head_idx], val_target, val_output[-head_idx]], dim=0)
    y_pred=val_output[0]

    if calib_name=='rcps':
        _lambda, _ = rcps(Y, "01", "hoeffding_bentkus", epsilon=float(alpha), delta=0.1,lambda_max=torch.tensor(0.1), stepsize=2e-04)
    else:
        opt_nums = np.random.choice(range(Y.shape[1]), Y.shape[1] // 2)
        Y_opt=Y[:,opt_nums]
        y_pred_opt=y_pred[opt_nums]
        Y_cal=Y[:,list(set(range(Y.shape[1]))-set(opt_nums))]
        y_pred_cal=y_pred[list(set(range(y_pred.shape[0]))-set(opt_nums))]
        _lambda, gain= kd_calibrate_train(len(opt_nums), y_pred_opt, Y_cal, Y_opt, epsilon=float(alpha))
    return _lambda, gain

def call_for_calibration_inf(lower, upper,pred,calib_name,_lambda,gain=None):
    if not calib_name in ['rcps', 'krcps']:
        return lower,upper
    if calib_name=='rcps':
        lambda_l, lambda_u = I(_lambda, lower, upper)
    else:
        k, _, m_val = loss_otsu(lower, upper, 10)
        _lambda_k_val = torch.matmul(m_val, _lambda) + gain
        lambda_l, lambda_u= I(_lambda_k_val, lower, upper)
    return pred-lambda_l, pred+lambda_u






