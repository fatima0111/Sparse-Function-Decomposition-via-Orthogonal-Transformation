import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from Utils.Function_utils import compute_hessian_orig_2d, compute_gradient_orig_2d,\
    noise_function
from Libs.Grid_Search import *
from Utils.Evaluation_utils import batches
from Libs.sbd_noise_robust import get_U
from Utils.Generation_utils import get_rank_svd
import copy


def func1():
    '''
    f(x_1, ..., x_7) = sin(2x_1)*x_7^3 + exp(
        -(x_1-1)^2)*(x_4+3) + cos(2*x_2)*sin(3*x_5) + x_6^2
    :return: f(x_1, ..., x_7)
    '''

    dim = 7
    blocks = [[0, 3, 6], [1, 4]]
    cops = [[0, 6], [0, 3], [1, 4]]
    max_components = 3
    t_sub_function_indices = [[1, 0], [4, 5], [2, 5]]
    t_num_sub_function_entries = [2, 2, 2]
    parameters = [[2, 3], [1, 1], [2, 3]]
    coeff = [7, 5, 10]
    torch.manual_seed(10)
    np.random.seed(10)
    v = torch.as_tensor(np.linalg.svd(np.random.rand(dim, dim))[0], dtype=torch.float64)
    ground_truth = {}
    ground_truth['n'] = dim
    ground_truth['max_components'] = max_components
    ground_truth['K'] = len(t_num_sub_function_entries)
    ground_truth['Blocs'] = blocks
    ground_truth['U'] = cops
    ground_truth['num_samples'] = int(1e2)
    ground_truth['t_sub_function_indices'] = [u for u in t_sub_function_indices]
    ground_truth['t_num_sub_function_entries'] = t_num_sub_function_entries
    ground_truth['parameters'] = parameters
    ground_truth['coeff'] = coeff
    ground_truth['v'] = v
    return ground_truth

def func2():
    '''
        f(x_1, ..., x_7) = sin(2x_1)*x_7^3 + exp(
            -(x_1-1)^2)*(x_4+3) + cos(2*x_2)*sin(3*x_5) + x_6^2
        :return: f(x_1, ..., x_7)
    '''

    dim = 7
    blocks = [[0, 1, 3, 6], [2, 4, 5]]
    cops = [[0, 6], [0, 3], [1, 6], [2, 4], [4, 5]]
    max_components = 4
    t_sub_function_indices = [[0, 0], [4, 2], [1, 2], [2, 1], [0, 0]]
    t_num_sub_function_entries = [2, 2, 2, 2, 2]
    parameters = [[1, 3], [1, 3], [1, 1], [2, 3], [1, 1]]
    coeff = [10, 5, 8, 12, 6]
    torch.manual_seed(10)
    np.random.seed(10)
    v = torch.as_tensor(np.linalg.svd(np.random.rand(dim, dim))[0], dtype=torch.float64)
    ground_truth = {}
    ground_truth['n'] = dim
    ground_truth['max_components'] = max_components
    ground_truth['K'] = len(t_num_sub_function_entries)
    ground_truth['Blocs'] = blocks
    ground_truth['U'] = cops
    ground_truth['num_samples'] = int(1e2)
    ground_truth['t_sub_function_indices'] = [u for u in t_sub_function_indices]
    ground_truth['t_num_sub_function_entries'] = t_num_sub_function_entries
    ground_truth['parameters'] = parameters
    ground_truth['coeff'] = coeff
    ground_truth['v'] = v
    return ground_truth

def set_gt(ground_truths):
    '''
    Generate data set from ground_truths information
    :param ground_truths: dictionary containing the information of the test function
    :return: extended dictionary containing ground_truth
    '''
    datas = {}
    for ind_g, ground_truth in enumerate(ground_truths):
        d = ground_truth['n']
        num_samples = int(1e2) * d
        torch.manual_seed(10)
        input_data = (2 * torch.rand(num_samples, d, dtype=torch.float64) - 1)
        hessianF_orig = lambda x_val: compute_hessian_orig_2d(
            x_val, ground_truth, return_coupling=True)
        hessian_o, U_ = hessianF_orig(x_val=input_data)
        copy_U_ = copy.deepcopy(U_)
        U = copy.deepcopy(ground_truth['U'])
        for u in U_:
            if u not in U and u[0] != u[1]:
                copy_U_.remove(u)
        ground_truth['U_true'] = copy_U_
        Num_non_zero_elements = []
        for bloc in ground_truth['Blocs']:
            hessian_b_o = hessian_o[:, bloc, :]
            hessian_b_o = hessian_b_o[:, :, bloc].abs().mean(dim=0)
            hessian_b_o[hessian_b_o != torch.clamp(hessian_b_o, 1e-6)] = 0
            Num_non_zero_elements.append('b_size: {} num_nzero: {}'.format(len(bloc), len(hessian_b_o.nonzero())))
        ground_truth['non_zero_block'] = Num_non_zero_elements
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
        Output['K'] = len(ground_truth['U'])
        Output['dim'] = d
        Output['n'] = num_samples
        Output['x_test'] = input_data
        Output['time'] = copy.deepcopy(tmp)
        Output['loss'] = copy.deepcopy(tmp)
        Output['M'] = copy.deepcopy(tmp)
        Output['R'] = copy.deepcopy(tmp)
        Output['groundtruth'] = ground_truth
        datas[str(ind_g)] = Output
    return datas

def train_model(datas, output_folder, N_epochs = int(5e4), run_man_opt=True, h_size=None):
    '''
    :param datas:
    :param output_folder:
    :param N_epochs:
    :param run_man_opt:
    :param h_size:
    :return:
    '''
    if run_man_opt:
        Compute_rotatation(datas, output_folder, N_epochs=N_epochs, run_man_opt=True, cov=None)
        Compute_rotatation(datas, output_folder, N_epochs=N_epochs,
                           run_man_opt=True, cov=0.5)
    else:
        h_sizes = [h_size] if h_size is not None else [1, 0.5, 0.25, 0.125]
        for h_size in h_sizes:
            Compute_rotatation(datas, output_folder, N_epochs=N_epochs, run_man_opt=run_man_opt,
                               cov=None, h_size=h_size)

            Compute_rotatation(datas, output_folder, N_epochs=N_epochs,
                               run_man_opt=run_man_opt, cov=0.5, h_size=h_size)
def Compute_rotatation(datas, output_folder, h_size=None,
               N_epochs = int(5e4), run_man_opt=True, cov=None):
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
    dtype = torch.float64
    for j in datas.keys():
        data = datas[j]
        ground_truth = data['groundtruth']
        dim = ground_truth['n']
        x_test = data['x_test']
        gradF_orig = compute_gradient_orig_2d(x_test, ground_truth)
        gradient = gradF_orig
        hessianF_orig = compute_hessian_orig_2d(x_test.clone(), ground_truth)
        if cov is not None:
            gradient_noise = noise_function(x_test.clone(), cov=cov, type='0', dtype=dtype)
            hessian_noise = noise_function(x_test.clone(), cov=cov, type='1', dtype=dtype)
            gradient =+ gradient_noise
            hessianF_orig += hessian_noise
        u1, d1, v1 = torch.svd(gradient)
        data['grad_U'] = u1
        supp = get_rank_svd(d1, u1.shape[0], eps=eps1)
        data['support'] = supp
        hessian_ = torch.matmul(torch.matmul(u1.T.to(device), hessianF_orig), u1.to(device))
        hessian = hessian_
        vec_hessian = hessian_.flatten(start_dim=1).T
        u2, d2, v2 = torch.svd(vec_hessian)
        rank_h = get_rank_svd(d2, u2.shape[0], eps=eps2)
        hessian_rank = u2.T[:rank_h].reshape(rank_h, dim, dim)
        hessian_rank = hessian_rank[:, :supp, :supp]
        data['rank_hessian'] = rank_h
        data['svd_basis'] = hessian_rank
        hessian_full = torch.clone(hessian)
        hessian = hessian[:, :supp, :supp]
        data['hessian'] = hessian_full
        P, BlockSizes = get_U(hessian_rank.to(torch.float64), epsilon1=1e-5, epsilon2=1e-5)
        data['SBD_GT'] = BlockSizes
        data['P'] = P
        print('BlockSizes alg: {} BlockSizes gt: {} '.format(BlockSizes,
                                                             data['groundtruth']['Blocs']))
        hessian_blocks = P @ hessian @ P.T
        hessian_rank_blocks = P @ hessian_rank @ P.T
        b_ad = 0
        b_ad_inner = 0
        for b in BlockSizes:
            if b > 1 and b <= 5:
                data['M']['gt']['Man_Opt'][b_ad_inner] = {}
                data['R']['gt']['Man_Opt'][b_ad_inner] = {}
                data['loss']['gt']['Man_Opt'][b_ad_inner] = {}

                data['M']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                data['R']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                data['loss']['gt']['Man_Opt_GS'][b_ad_inner] = {}

                data['M']['gt']['Grid_search'][b_ad_inner] = {}
                data['R']['gt']['Grid_search'][b_ad_inner] = {}
                data['loss']['gt']['Grid_search'][b_ad_inner] = {}
                hessian_rank_b = hessian_rank_blocks[:, b_ad:b_ad + b, b_ad:b_ad + b]
                hessian_b = hessian_blocks[:, b_ad:b_ad + b, b_ad:b_ad + b]

                if run_man_opt:
                    result_man_opt = run_MO_Block(
                        hessian_rank_b, optimizer_method=Method.Manifold_Opt, N_epochs=N_epochs)
                    Bs_man_opt, losses_man_opt, times_man_opt = result_man_opt

                    data['M']['gt']['Man_Opt'][b_ad_inner]['la'] = (
                            Bs_man_opt[0] @ hessian_b @ Bs_man_opt[0].T).abs().mean(dim=0)
                    data['M']['gt']['Man_Opt'][b_ad_inner]['re'] = (
                            Bs_man_opt[1] @ hessian_b @ Bs_man_opt[1].T).abs().mean(dim=0)

                    data['R']['gt']['Man_Opt'][b_ad_inner]['la'] = Bs_man_opt[0]
                    data['R']['gt']['Man_Opt'][b_ad_inner]['re'] = Bs_man_opt[1]

                    data['loss']['gt']['Man_Opt'][b_ad_inner]['la'] = losses_man_opt[0]
                    data['loss']['gt']['Man_Opt'][b_ad_inner]['re'] = losses_man_opt[1]

                else:
                    batch_h = batches[b][h_size]
                    result_man_opt_gs, result_grid_search, time_grid_search = run_MO_Block(
                        hessian_rank_b, optimizer_method=Method.Manifold_Opt_GS,
                        h=h_size, batch_h=batch_h, N_epochs=N_epochs)
                    Bs_man_opt_gs, losses_man_opt_gs, times_man_opt_gs = result_man_opt_gs

                    data['M']['gt']['Grid_search'][b_ad_inner] = (
                            result_grid_search[0] @ hessian_b @ result_grid_search[0].T).abs().mean(dim=0)
                    data['M']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = (
                            Bs_man_opt_gs[0] @ hessian_b @ Bs_man_opt_gs[0].T).abs().mean(dim=0)
                    data['M']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = (
                            Bs_man_opt_gs[1] @ hessian_b @ Bs_man_opt_gs[1].T).abs().mean(dim=0)

                    data['R']['gt']['Grid_search'][b_ad_inner] = result_grid_search[0]
                    data['R']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = Bs_man_opt_gs[0]
                    data['R']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = Bs_man_opt_gs[1]

                    data['loss']['gt']['Grid_search'][b_ad_inner] = result_grid_search[1]
                    data['loss']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = losses_man_opt_gs[0]
                    data['loss']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = losses_man_opt_gs[1]

                b_ad_inner += 1
            b_ad += b
            suff_ = '_cov_{}'.format(cov) if cov is not None else ''
            if not run_man_opt:
                with open('{}/Test_anova{}_h_size_{}.json'.format(output_folder, suff_, h_size), 'w') as convert_file:
                    json.dump(datas, convert_file, cls=NumpyEncoder)
            else:
                with open('{}/Test_anova{}_mo.json'.format(output_folder, suff_), 'w') as convert_file:
                    json.dump(datas, convert_file, cls=NumpyEncoder)
    return datas



if __name__ == '__main__':

    output_folder = '/homes/numerik/fatimaba/store/Github/trafo_nova/Anova_AE/Output_files'
    cov = None
    run_mo_gs = True
    tr_model = False
    evaluate = False
    if tr_model:
        datas = set_gt([func1(), func2()])
        train_model(datas, output_folder, h_size=1, N_epochs=int(5e4), run_man_opt=run_mo_gs)
        train_model(datas, output_folder, N_epochs=int(5e4), run_man_opt=not run_mo_gs)
