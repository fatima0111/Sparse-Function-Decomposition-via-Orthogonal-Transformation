
import copy
import json
import torch
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda')
    gdtype = torch.float64
else:
    device = torch.device('cpu')
    gdtype = torch.float64
from Utils.Function_utils import compute_hessian_orig_2d, compute_gradient_orig_2d,\
    noise_function
from Libs.Grid_Search import *
from Utils.Generation_utils import batches_random_functions
from Libs.sbd_noise_robust import get_U
from Utils.Generation_utils import get_rank_svd
from Utils.Evaluation_utils import Init_Method
from Generation.Generate_Datasets.Generate_ANOVA_sparse_functions import *
from Utils.Evaluation_utils import NumpyEncoder
from os.path import dirname, abspath

def set_gt(ground_truths):
    '''
    Generate dataset from ground_truths information
    :param ground_truths: dictionary containing the information of the test function
    :return: extended dictionary containing ground_truth
    '''
    datas = {}
    for ind_g, ground_truth in enumerate(ground_truths):
        d = ground_truth['d']
        num_samples = ground_truth['N']
        torch.manual_seed(10)
        input_data = (2 * torch.rand(num_samples, d, dtype=torch.float64) - 1)
        hessianF_orig = lambda x_val: compute_hessian_orig_2d(
            x_val, ground_truth, return_coupling=True)
        hessian_o, J_ = hessianF_orig(x_val=input_data)
        copy_J_ = copy.deepcopy(J_)
        J = copy.deepcopy(ground_truth['J'])
        for u in J_:
            if u not in J and u[0] != u[1]:
                copy_J_.remove(u)
        ground_truth['J_true'] = copy_J_
        Num_non_zero_elements = []
        for bloc in ground_truth['blocs']:
            hessian_b_o = hessian_o[:, bloc, :]
            hessian_b_o = hessian_b_o[:, :, bloc].abs().mean(dim=0)
            hessian_b_o[hessian_b_o != torch.clamp(hessian_b_o, 1e-6)] = 0
            Num_non_zero_elements.append('b_size: {} num_nzero: {}'.format(len(bloc), len(hessian_b_o.nonzero())))
        ground_truth['non_zero_bloc'] = Num_non_zero_elements
        tmp1 = {
            'Grid_search': {},
            'Man_Opt': {
                'la': {},
                're': {}
            },
            'Man_Opt_GS': {
                'la': {},
                're': {}
            }
        }
        tmp = {'gt': copy.deepcopy(tmp1), 'noise': copy.deepcopy(tmp1)}
        Output = {}

        Output['x_test'] = input_data
        Output['time'] = copy.deepcopy(tmp)
        Output['loss'] = copy.deepcopy(tmp)
        #Output['M'] = copy.deepcopy(tmp)
        Output['U'] = copy.deepcopy(tmp)
        Output['groundtruth'] = ground_truth
        datas[str(ind_g)] = Output
    return datas

def Compute_rotatation(datas, output_folder, h_size=None,
               N_epochs = int(5e4), init_method=Init_Method.RI, cov=None):
    '''
    :param datas:
    :param output_folder:
    :param h_size:
    :param N_epochs:
    :param run_man_opt:
    :param cov:
    :return:
    '''

    eps1 = 3 if cov is not None else 3
    eps2 = 2 if cov is not None else 3

    batches = batches_random_functions
    for j in datas.keys():
        data = datas[j]
        ground_truth = data['groundtruth']
        dim = ground_truth['d']
        x_test = data['x_test']
        gradF_orig = compute_gradient_orig_2d(x_test, ground_truth)
        gradient = gradF_orig
        hessianF_orig = compute_hessian_orig_2d(x_test.clone(), ground_truth)
        if cov is not None:
            gradient_noise = noise_function(x_test.clone(), cov=cov, type='0', dtype=gdtype)
            hessian_noise = noise_function(x_test.clone(), cov=cov, type='1', dtype=gdtype)
            gradient += gradient_noise
            hessianF_orig += hessian_noise
        u1, d1, v1 = torch.svd(gradient)
        data['U_svd'] = u1
        supp = get_rank_svd(d1, u1.shape[0], eps=eps1)
        data['rank_grad'] = supp
        hessian_ = torch.matmul(torch.matmul(u1.T.to(device), hessianF_orig), u1.to(device))
        hessian = hessian_
        vec_hessian = hessian_.flatten(start_dim=1).T
        u2, d2, v2 = torch.svd(vec_hessian)
        rank_h = get_rank_svd(d2, u2.shape[0], eps=eps2)
        hessian_rank = u2.T[:rank_h].reshape(rank_h, dim, dim)
        hessian_rank = hessian_rank[:, :supp, :supp]
        data['rank_hessian'] = rank_h
        data['hessian_basis'] = hessian_rank
        hessian_full = torch.clone(hessian)
        hessian = hessian[:, :supp, :supp]
        data['hessian'] = hessian_full
        U_bloc, blocSizes = get_U(hessian_rank.to(torch.float64), epsilon1=1e-5, epsilon2=1e-5)
        data['blocs_alg'] = blocSizes
        data['U_bloc'] = U_bloc
        print('blocSizes alg: {} blocSizes gt: {} '.format(blocSizes,
                                                             data['groundtruth']['blocs']))
        hessian_blocs = U_bloc @ hessian @ U_bloc.T
        hessian_rank_blocs = U_bloc @ hessian_rank @ U_bloc.T

        tmp = {
            'Grid_search': {},
            'Man_Opt_RI': {
                'la': {},
                'rgd': {}
            },
            'Man_Opt_GS': {
                'la': {},
                'rgd': {}
            }
        }
        b_ad = 0
        b_ad_inner = 0
        data['loss'] = copy.deepcopy(tmp)
        data['U'] = copy.deepcopy(tmp)
        for b in blocSizes:
            if b > 1 and b <= 5:
                #data['M']['clean']['Man_Opt_RI'][b_ad_inner] = {}
                data['U']['Man_Opt_RI'][b_ad_inner] = {}
                data['loss']['Man_Opt_RI'][b_ad_inner] = {}

                #data['M']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                data['U']['Man_Opt_GS'][b_ad_inner] = {}
                data['loss']['Man_Opt_GS'][b_ad_inner] = {}

                #data['M']['gt']['Grid_search'][b_ad_inner] = {}
                data['U']['Grid_search'][b_ad_inner] = {}
                data['loss']['Grid_search'][b_ad_inner] = {}
                hessian_rank_b = hessian_rank_blocs[:, b_ad:b_ad + b, b_ad:b_ad + b]
                #hessian_b = hessian_blocs[:, b_ad:b_ad + b, b_ad:b_ad + b]

                if init_method == Init_Method.RI:
                    result_man_opt = run_MO_bloc(
                        hessian_rank_b, optimizer_method=Method.Manifold_Opt, N_epochs=N_epochs)
                    Us_RI, losses_RI, _ = result_man_opt

                    #data['M']['gt']['Man_Opt'][b_ad_inner]['la'] = (
                    #        Bs_man_opt[0] @ hessian_b @ Bs_man_opt[0].T).abs().mean(dim=0)
                    #data['M']['gt']['Man_Opt'][b_ad_inner]['rgd'] = (
                    #        Bs_man_opt[1] @ hessian_b @ Bs_man_opt[1].T).abs().mean(dim=0)

                    data['U']['Man_Opt_RI'][b_ad_inner]['la'] = Us_RI[0]
                    data['U']['Man_Opt_RI'][b_ad_inner]['rgd'] = Us_RI[1]

                    data['loss']['Man_Opt_RI'][b_ad_inner]['la'] = losses_RI[0]
                    data['loss']['Man_Opt_RI'][b_ad_inner]['rgd'] = losses_RI[1]

                else:
                    batch_h = batches[b][h_size]
                    result_man_opt_gs, result_grid_search, time_grid_search = run_MO_bloc(
                        hessian_rank_b, optimizer_method=Method.Manifold_Opt_GS,
                        h=h_size, batch_h=batch_h, N_epochs=N_epochs)
                    Us_GS, losses_GS, _ = result_man_opt_gs

                    #data['M']['gt']['Grid_search'][b_ad_inner] = (
                    #        result_grid_search[0] @ hessian_b @ result_grid_search[0].T).abs().mean(dim=0)
                    #data['M']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = (
                    #        Bs_man_opt_gs[0] @ hessian_b @ Bs_man_opt_gs[0].T).abs().mean(dim=0)
                    #data['M']['gt']['Man_Opt_GS'][b_ad_inner]['rgd'] = (
                    #        Bs_man_opt_gs[1] @ hessian_b @ Bs_man_opt_gs[1].T).abs().mean(dim=0)

                    data['U']['Grid_search'][b_ad_inner] = result_grid_search[0]
                    data['U']['Man_Opt_GS'][b_ad_inner]['la'] = Us_GS[0]
                    data['U']['Man_Opt_GS'][b_ad_inner]['rgd'] = Us_GS[1]

                    data['loss']['Grid_search'][b_ad_inner] = result_grid_search[1]
                    data['loss']['Man_Opt_GS'][b_ad_inner]['la'] = losses_GS[0]
                    data['loss']['Man_Opt_GS'][b_ad_inner]['rgd'] = losses_GS[1]

                b_ad_inner += 1
            b_ad += b
            suff_ = '_cov_{}'.format(cov) if cov is not None else ''
            if init_method == Init_Method.GS:
                with open('{}/Test_ANOVA{}_h_size_{}.json'.format(output_folder, suff_, h_size), 'w') as convert_file:
                    json.dump(datas, convert_file, cls=NumpyEncoder)
            else:
                with open('{}/Test_ANOVA{}_MO_RI.json'.format(output_folder, suff_), 'w') as convert_file:
                    json.dump(datas, convert_file, cls=NumpyEncoder)
    return datas

if __name__ == '__main__':
    output_folder = dirname(dirname(abspath(__file__))) + '/Output/Output_algorithms/ANOVA_sparse_functions'
    init_method = Init_Method.GS
    covs = [None, 0.5]
    h_size = 1.0
    batches = batches_random_functions
    for cov in covs:
        datas = set_gt([func1(), func2()])
        Compute_rotatation(datas, output_folder, h_size=h_size, N_epochs=int(5e4), init_method=init_method, cov=cov)
        Compute_rotatation(datas, output_folder, h_size=h_size, N_epochs=int(5e4), init_method=init_method, cov=cov)
