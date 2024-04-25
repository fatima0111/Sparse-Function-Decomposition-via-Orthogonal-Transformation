import torch
import numpy as np
import math
import json
from enum import Enum
from Libs.Grid_Search import Method

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    #device = torch.device('cuda')
#else:
    #device = torch.device('cpu')

batches = {
    2: {1: math.pi,
        1 / 2: math.pi,
        1 / 4: math.pi,
        1 / 8: math.pi},
    3: {1: math.pi / 2,
        1 / 2: math.pi / 2,
        1 / 4: math.pi / 2,
        1 / 8: math.pi / 2},
    4: {1: math.pi / 2,
        1 / 2: math.pi / 2,
        1 / 4: math.pi / 2,
        1 / 8: 1.0}
}

class Init_Method(Enum):
    RI = 1
    GS = 2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)

def get_total_Rot(data, j, init_method=Init_Method.RI):
    '''

    :param data:
    :param j:
    :param random_init:
    :return:
    '''
    ground_truth = data[j]['groundtruth']
    ground_truth['v'] = torch.as_tensor(ground_truth['v'])
    d = data[j]['dim']
    BlockSizes = data[j]['SBD_GT']
    supp = int(data[j]['support'])
    U1 = torch.as_tensor(data[j]['grad_U'])
    P = torch.eye(d)
    P[:supp, :supp] = torch.as_tensor(data[j]['P'])
    U_La_ = torch.eye(d)
    U_Rgd_ = torch.eye(d)
    b_ad = 0
    b_ad_inner = 0
    for b in BlockSizes:
        if b > 1 and b <= 5:
            b_ad_inner_ = str(b_ad_inner)
            if init_method == Init_Method.RI:
                U_La_[b_ad:b_ad + b, b_ad:b_ad + b] = torch.as_tensor(data[j]['R']['clean']['Man_Opt_RI'][b_ad_inner_]['la'])

                U_Rgd_[b_ad:b_ad + b, b_ad:b_ad + b] = torch.as_tensor(data[j]['R']['clean']['Man_Opt_RI'][b_ad_inner_]['rgd'])
            else:
                U_La_[b_ad:b_ad + b, b_ad:b_ad + b] = torch.as_tensor(
                    data[j]['R']['clean']['Man_Opt_GS'][b_ad_inner_]['la'])
                U_Rgd_[b_ad:b_ad + b, b_ad:b_ad + b] = torch.as_tensor(
                    data[j]['R']['clean']['Man_Opt_GS'][b_ad_inner_]['rgd'])
            b_ad_inner += 1
        b_ad += b
    return U_La_ @ P @ U1.T, U_Rgd_ @ P @ U1.T


def compute_hessian_rotmatrix(data, method, noisy_data=False, noisy_rot=False,
                              basis_hess=False, p=1):
    '''
    :param data:
    :param method:
    :param noisy_data:
    :param noisy_rot:
    :param basis_hess:
    :param p:
    :return:
    '''
    key_clean_noisy = 'clean' if not noisy_data else 'noisy'
    key_hessian = 'svd_basis' if basis_hess else 'hessian'
    hessian = torch.as_tensor(data[key_hessian][key_clean_noisy], dtype=torch.float64)
    method_name = 'Man_Opt_RI' if method == Method.Manifold_Opt else 'Man_Opt_GS' if method == Method.Manifold_Opt_GS else 'Grid_search'
    clean_noisy_key = 'noisy' if noisy_rot else 'clean'
    U_rgd = torch.as_tensor(data['R'][clean_noisy_key][method_name]['rgd'], dtype=torch.float64)
    U_la = torch.as_tensor(data['R'][clean_noisy_key][method_name]['la'], dtype=torch.float64)
    if p == 1:
        matrix_rgd = (U_rgd @ hessian @ U_rgd.T).abs().mean(dim=0)
        matrix_la = (U_la @ hessian @ U_la.T).abs().mean(dim=0)
        return matrix_rgd, matrix_la
    elif p == 2:
        matrix_rgd = ((U_rgd @ hessian @ U_rgd.T).abs() ** 2).mean(dim=0).sqrt()
        matrix_la = ((U_la @ hessian @ U_la.T).abs() ** 2).mean(dim=0).sqrt()
        return matrix_rgd, matrix_la
    elif p == math.inf:
        matrix_rgd = (U_rgd @ hessian @ U_rgd.T).abs().max(dim=0)
        matrix_la = (U_la @ hessian @ U_la.T).abs().max(dim=0)
        return matrix_rgd, matrix_la
    else:
        print('\n p is not defined')
        exit(1)

def get_len_loss(names, out_dir):
    '''
    :param names:
    :param out_dir:
    :return:
    '''
    for name in names:
        with open('{}/{}'.format(out_dir, name), 'r') as convert_file:
            data = json.load(convert_file)
            for j in data.keys():
                if data[j]['h_size'] == 1 / 2:
                    if len(data[j]['loss']['clean']['Man_Opt_RI']['la']) > max_loss_man_opt_2:
                        max_loss_man_opt_2 = len(data[j]['loss']['clean']['Man_Opt_RI']['la'])
                    if len(data[j]['loss']['clean']['Man_Opt_RI']['rgd']) > max_loss_man_opt_2:
                        max_loss_man_opt_2 = len(data[j]['loss']['clean']['Man_Opt_RI']['rgd'])

                    if len(data[j]['loss']['noise']['Man_Opt_RI']['la']) > max_loss_man_opt_2_noise:
                        max_loss_man_opt_2_noise = len(
                            data[j]['loss']['noise']['Man_Opt_RI']['la'])
                    if len(data[j]['loss']['noise']['Man_Opt_RI']['rgd']) > max_loss_man_opt_2_noise:
                        max_loss_man_opt_2_noise = len(
                            data[j]['loss']['noise']['Man_Opt_RI']['rgd'])
                if len(data[j]['loss']['clean']['Man_Opt_GS']['la']) > max_loss_man_opt_2:
                    max_loss_man_opt_2 = len(data[j]['loss']['clean']['Man_Opt_GS']['la'])

                if len(data[j]['loss']['clean']['Man_Opt_GS']['rgd']) > max_loss_man_opt_2:
                    max_loss_man_opt_2 = len(data[j]['loss']['clean']['Man_Opt_GS']['rgd'])

                if len(data[j]['loss']['noise']['Man_Opt_GS']['la']) > max_loss_man_opt_2_noise:
                    max_loss_man_opt_2_noise = len(data[j]['loss']['noise']['Man_Opt_GS']['la'])

                if len(data[j]['loss']['clean']['Man_Opt_GS']['rgd']) > max_loss_man_opt_2_noise:
                    max_loss_man_opt_2_noise = len(data[j]['loss']['noise']['Man_Opt_GS']['rgd'])
        print('max_loss_man_opt_2: ', max_loss_man_opt_2)
        print('max_loss_man_opt_re: ', max_loss_man_opt_2_noise)
        return max_loss_man_opt_2, max_loss_man_opt_2_noise


def loss_function12(rot, hessian, eps=0):
    '''
    :param rot:
    :param hessian:
    :param eps:
    :return:
    '''
    if rot.dtype == torch.float32:
        rot = rot.to(torch.float64)
    if hessian.dtype == torch.float32:
        hessian = hessian.to(torch.float64)

    if rot.ndim == 2:
        res = torch.matmul(torch.matmul(rot, hessian), rot.t())
        res2 = ((res.abs() ** 2).mean(dim=0) + eps).sqrt()
        loss = res2.sum()
        return loss
    elif rot.ndim == 4:
        mult = torch.matmul(rot, hessian)
        res = torch.matmul(mult, rot.permute([0, 1, 3, 2]))
        res2 = ((res.abs() ** 2).mean(dim=1) + eps).sqrt()
        loss = res2.sum(dim=[1, 2])
        return loss


def loss_function_sparse(rot, hessian):
    '''
    :param rot:
    :param hessian:
    :param upper:
    :return:
    '''
    if rot.dtype == torch.float32:
        rot = rot.to(torch.float64)
    if hessian.dtype == torch.float32:
        hessian = hessian.to(torch.float64)

    res = torch.matmul(torch.matmul(rot, hessian), rot.t())
    res2 = (res.abs() ** 2).mean(dim=0) ** .25
    loss = res2.sum() ** 2
    return loss

