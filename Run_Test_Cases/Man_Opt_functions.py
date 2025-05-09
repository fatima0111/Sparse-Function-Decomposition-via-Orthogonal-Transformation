import argparse
import copy
from os.path import dirname, abspath
import torch
import json
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda')
    gdtype = torch.float64
else:
    device = torch.device('cpu')
    gdtype = torch.float64

from Utils.Function_utils import compute_function, compute_hessian_orig_2d, \
    compute_gradient_orig_2d, noise_function
from Libs.Grid_Search import *
from Utils.Generation_utils import get_rank_svd, batches_random_functions
from Libs.sbd_noise_robust import get_U
from Utils.Evaluation_utils import NumpyEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default='/work/ba/output',
                        type=str, help='Path of the Output folder')
    parser.add_argument('--start_N_run', default=0, type=int,
                        help='Number of runs')
    parser.add_argument('--N_run', default=50, type=int,
                        help='Number of runs')
    parser.add_argument('--N_epochs', default=int(5e4) + 1, type=int,
                        help='Number of epochs')
    parser.add_argument('--h_size', default=1, type=float,
                        help='Inner step_size Grid_seach')
    parser.add_argument('--run_man_opt', default=False, type=bool,
                        help='Var to run random initialization manifold optimization')
    parser.add_argument('--cov', default=None, type=float,
                        help='standard deviation of the noise function')
    parser.add_argument('--test_cases', default='Test_functions_M_50.json', type=str,
                        help='filename where the test_cases are saved')
    parser.add_argument('--opt_method', default='both', type=str,
                        help='Manifold optimization method: 1.RiemannianSGD, 2.LandingSGD, default:both methods')
    parser.add_argument('--learning_rate', default=5e-4, type=int,
                        help='learning rate for the manifold optimization method')
    args = parser.parse_args()

    h_size = args.h_size
    # $ conda activate /work/ba/anova
    start_N_run = args.start_N_run
    N_run = args.N_run
    N_epochs = args.N_epochs
    run_man_opt = args.run_man_opt
    in_dir = dirname(dirname(abspath(__file__))) + '/Dataset'
    cov = args.cov
    test_cases = args.test_cases
    opt_method = args.opt_method
    learning_rate = args.learning_rate

    batches = batches_random_functions
    with open('{}/{}'.format(in_dir, test_cases)) as convert_file:
        data = copy.deepcopy(json.load(convert_file))
        h_ft = torch.zeros((50, 2))
        g_ft = torch.zeros((50, 2))
        for j in range(start_N_run, N_run):
            j = str(j)
            d = data[j]['d']
            R = torch.as_tensor(data[j]['groundtruth']['R'], dtype=gdtype)
            blocs = data[j]['groundtruth']['blocs']
            num_samples = data[j]['groundtruth']['N']
            ground_truth = data[j]['groundtruth']
            ground_truth['R'] = R
            means = 0 if cov is not None else None
            x_test = data[j]['x_test'] = torch.as_tensor(data[j]['x_test'], dtype=gdtype)
            target_test = compute_function(x_test, ground_truth=ground_truth)
            eps1 = 3 if cov is not None else 3
            eps2 = 2 if cov is not None else 3
            gradF_orig = compute_gradient_orig_2d(x_test, ground_truth)
            gradient = gradF_orig
            hessianF_orig = compute_hessian_orig_2d(x_test.clone(), ground_truth)
            if cov is not None:
                gradient_noise = noise_function(x_test.clone(), cov=cov, type='0', dtype=gdtype)
                hessian_noise = noise_function(x_test.clone(), cov=cov, type='1', dtype=gdtype)
                gradient += gradient_noise
            u1, d1, v1 = torch.svd(gradient)
            data[j]['U_svd'] = u1
            supp = get_rank_svd(d1, u1.shape[0], eps=eps1)
            data[j]['rank_grad'] = supp

            x_hessian = x_test.clone()
            if cov is not None:
                hessianF_orig += hessian_noise

            hessian_ = torch.matmul(torch.matmul(u1.T.to(device), hessianF_orig), u1.to(device))
            hessian = hessian_
            vec_hessian = hessian_.flatten(start_dim=1).T
            u2, d2, v2 = torch.svd(vec_hessian)
            rank_h = get_rank_svd(d2, u2.shape[0], eps=eps2)
            hessian_rank = u2.T[:rank_h].reshape(rank_h, d, d)
            hessian_rank = hessian_rank[:, :supp, :supp]
            data[j]['rank_hessian'] = rank_h
            data[j]['hessian_basis'] = hessian_rank
            hessian_full = torch.clone(hessian)
            hessian = hessian[:, :supp, :supp]
            data[j]['hessian'] = hessian_full

            U_bloc, blocSizes = get_U(hessian_rank.to(gdtype), epsilon1=1e-5, epsilon2=1e-5)
            data[j]['blocs_alg'] = blocSizes
            data[j]['U_bloc'] = U_bloc
            print('blocSizes alg: {} blocSizes gt: {} '.format(
                blocSizes, data[j]['groundtruth']['blocs']))
            hessian_blocs = U_bloc @ hessian @ U_bloc.T
            hessian_rank_blocs = U_bloc @ hessian_rank @ U_bloc.T
            b_ad = 0
            b_ad_inner = 0
            for b in blocSizes:
                if b > 1 and b <= 5:
                    data[j]['U']['Man_Opt_RI'][b_ad_inner] = {}
                    data[j]['loss']['Man_Opt_RI'][b_ad_inner] = {}

                    data[j]['U']['Man_Opt_GS'][b_ad_inner] = {}
                    data[j]['loss']['Man_Opt_GS'][b_ad_inner] = {}

                    data[j]['U']['Grid_search'][b_ad_inner] = {}
                    data[j]['loss']['Grid_search'][b_ad_inner] = {}
                    hessian_rank_b = hessian_rank_blocs[:, b_ad:b_ad + b, b_ad:b_ad + b]
                    hessian_b = hessian_blocs[:, b_ad:b_ad + b, b_ad:b_ad + b]
                    batch_h = batches[b][h_size]
                    if h_size == 1/2 or run_man_opt:
                        result_man_opt = run_Man_Opt(
                            hessian_rank_b, optimizer_method=Method.Manifold_Opt,
                            h=h_size, batch_h=batch_h, N_epochs=N_epochs,
                            opt_method=opt_method, learning_rate=learning_rate)
                        Us_RI, losses_RI, _ = result_man_opt

                        data[j]['U']['Man_Opt_RI'][b_ad_inner]['la'] = Us_RI[0]
                        data[j]['U']['Man_Opt_RI'][b_ad_inner]['rgd'] = Us_RI[1]

                        data[j]['loss']['Man_Opt_RI'][b_ad_inner]['la'] = losses_RI[0]
                        data[j]['loss']['Man_Opt_RI'][b_ad_inner]['rgd'] = losses_RI[1]

                    result_man_opt_gs, result_grid_search = run_Man_Opt(hessian_rank_b, optimizer_method=Method.Manifold_Opt_GS,
                        h=h_size, batch_h=batch_h, N_epochs=N_epochs, opt_method=opt_method, learning_rate=learning_rate)
                    Us_GS, losses_GS, _ = result_man_opt_gs

                    data[j]['U']['Grid_search'][b_ad_inner] = result_grid_search[0]
                    data[j]['U']['Man_Opt_GS'][b_ad_inner]['la'] = Us_GS[0]
                    data[j]['U']['Man_Opt_GS'][b_ad_inner]['rgd'] = Us_GS[1]

                    data[j]['loss']['Grid_search'][b_ad_inner] = result_grid_search[1]
                    data[j]['loss']['Man_Opt_GS'][b_ad_inner]['la'] = losses_GS[0]
                    data[j]['loss']['Man_Opt_GS'][b_ad_inner]['rgd'] = losses_GS[1]

                    b_ad_inner += 1
                b_ad += b
                suff_ = '_cov_{}'.format(cov) if cov is not None else ''
                with open('{}/function_Man_opt_grid_search{}{}_h_{}_M_50.json'.format(
                        args.output_folder, N_run, suff_, args.h_size), 'w') as convert_file:
                    json.dump(data, convert_file, cls=NumpyEncoder)









