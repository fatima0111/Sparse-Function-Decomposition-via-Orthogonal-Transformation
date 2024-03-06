import argparse
import copy
from os.path import dirname, abspath
import os
import sys

import torch

sys.path.append('/homes/math/ba/trafo_nova/')
sys.path.append('//')
from Anova_AE.Libs.Utils import compute_gradient_autograd, \
    train_model, random_function, compute_hessian, compute_hessian_orig_2d, \
    compute_gradient_orig_2d, noise_function

from Anova_NN.Model_class import SparseAdditiveModel
from Anova_AE.Libs.Grid_Search import *
from Anova_AE.Libs.roots import get_rank_svd, get_U
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default='/work/ba/output',
                        type=str, help='Path of the Output folder')  #
    #'/homes/numerik/fatimaba/store/Github/trafo_nova/Anova_AE/Output_files'
    parser.add_argument('--start_N_run', default=0, type=int,
                        help='Number of runs')
    parser.add_argument('--N_run', default=50, type=int,
                        help='Number of runs')
    parser.add_argument('--N_epochs', default=int(5e4) + 1, type=int,
                        help='Number of epochs')#
    parser.add_argument('--h_size', default=1, type=float,
                        help='Inner step_size Grid_seach')
    parser.add_argument('--run_man_opt', default=False, type=bool,
                        help='Var to run random initialization manifold optimization')
    parser.add_argument('--NN_train', default=False, type=bool,
                        help='Wether using a NN to compute the gradient and hessian'
                             'or directly use the groundtruth')
    parser.add_argument('--cov', default=None, type=float,
                        help='standard deviation of the noise function')
    parser.add_argument('--test_cases', default='Test_functions_N_50_coeff.json', type=str,
                        help='filename where the test_cases are saved')
    args = parser.parse_args()

    h_size = args.h_size
    # $ conda activate /work/ba/anova
    start_N_run = args.start_N_run
    N_run = args.N_run  # 100
    NN_train = args.NN_train
    N_epochs = args.N_epochs
    root = '/store/steinitz/datastore/fatimaba/Github/trafo_nova/Anova_AE'#os.getcwd()
    root_NN = root + '/Output_NN/'
    root_files = root + '/Output_files'
    existing_model = True
    run_man_opt = args.run_man_opt
    in_dir = dirname(dirname(abspath(__file__))) + '/Test_Cases_Man_Opt_GS'
    cov = args.cov
    test_cases = args.test_cases
    # os.path.dirname(os.getcwd())+'/Anova_AE/Test_Cases_Man_Opt_GS'

    batches = {
        2: {3: math.pi,
            1: math.pi,
            1/2: math.pi,
            1/4: math.pi,
            1/8: math.pi},
        3: {3: math.pi,
            1: math.pi / 2,
            1 / 2: math.pi / 2,
            1 / 4: math.pi / 2,
            1 / 8: math.pi / 2},
        4: {3: math.pi,
            1: math.pi / 2,
            1 / 2: math.pi / 2,
            1 / 4: math.pi / 2,
            1 / 8: 1.0}
    }
    pen = 1
    lambd = 0.00001 if pen == 2 else 0.01
    N_test_c = 50
    dtype = torch.float64
    with open('{}/{}'.format(in_dir, test_cases)) as convert_file:
        data = copy.deepcopy(json.load(convert_file))
        h_ft = torch.zeros((50, 2))
        g_ft = torch.zeros((50, 2))
        for j in range(start_N_run, N_run):
            j = str(j)
            dim = data[j]['dim']
            v = torch.as_tensor(data[j]['groundtruth']['v'], dtype=torch.float64)
            blocks = data[j]['groundtruth']['Blocs']
            num_samples = data[j]['groundtruth']['n']
            ground_truth = data[j]['groundtruth']
            ground_truth['v'] = v
            means = 0 if cov is not None else None

            x_test = data[j]['x_test'] = torch.as_tensor(data[j]['x_test'], dtype=torch.float64)
            target_test = random_function(x_test, ground_truth=ground_truth)

            if NN_train:
                NN_K = 1
                N_train = int(25e3) * dim
                x = (2 * torch.rand(N_train, dim) - 1)
                target = random_function(x, ground_truth)
                sigma = 1
                if not existing_model:
                    model_params, my_print = train_model(x, target, NN_K,
                                                         batch_size=N_train // 2,
                                                         sigma=sigma)
                    torch.save(model_params,
                               root_NN + 'model_{}_lambd_{}_pen_{}_N{}_{}d.pt'.format(j, lambd, pen, N_train,
                                                                                      data[j]['groundtruth']['max_components']))
                    root_NN_training = root + '/Output_NN_training'
                    if not os.path.isdir(root_files):
                        os.mkdir(root_files)
                    myfile = open('{}/f_n_model_{}_lambd_{}_pen_{}_NN.txt'.format(root_NN_training, j, lambd, pen), "w")
                    myfile.write(my_print)
                    myfile.close()
                else:
                    path = root_NN + 'model_{}_lambd_{}_pen_{}_N{}_{}d.pt'.format(j, lambd, pen, N_train,
                                                                                  data[j]['groundtruth'][
                                                                                      'max_components'])
                    model_params = torch.load(path, map_location=device)
                    model = SparseAdditiveModel(dim, NN_K, setting=1)
                    model.load_state_dict(model_params)
                    model.eval()

                    model.set_sigma(sigma)
                    nn_loss = (((model(x_test.clone()).squeeze() - target_test.squeeze().to(
                        device)) ** 2) / model.sigma ** 2).mean()
                    print("\n NN_loss {}".format(nn_loss), model.sigma, model.sub_networks[0].sigma)
                    model.to(device)
            eps1 = 3 if (NN_train or cov is not None) else 3
            eps2 = 2 if (NN_train or cov is not None) else 3
            y_test = random_function(x_test, ground_truth)
            gradF_orig = compute_gradient_autograd(x_test.clone(), model=model
                                                   ) if NN_train else compute_gradient_orig_2d(
                x_test, ground_truth)
            gradient = gradF_orig
            hessianF_orig = compute_hessian(x_test.clone(), model=model, dtype=dtype
                                            ) if NN_train else compute_hessian_orig_2d(
                x_test.clone(), ground_truth)
            if cov is not None:
                gradient_noise = noise_function(x_test.clone(), cov=cov, type='0', dtype=dtype)
                hessian_noise = noise_function(x_test.clone(), cov=cov, type='1', dtype=dtype)
                gradient += gradient_noise
            u1, d1, v1 = torch.svd(gradient)
            data[j]['grad_U'] = u1
            supp = get_rank_svd(d1, u1.shape[0], eps=eps1)
            data[j]['support'] = supp

            x_hessian = x_test.clone()
            if cov is not None:
                hessianF_orig += hessian_noise

            hessian_ = torch.matmul(torch.matmul(u1.T.to(device), hessianF_orig), u1.to(device))

            hessian = hessian_
            vec_hessian = hessian_.flatten(start_dim=1).T
            u2, d2, v2 = torch.svd(vec_hessian)
            rank_h = get_rank_svd(d2, u2.shape[0], eps=eps2)
            hessian_rank = u2.T[:rank_h].reshape(rank_h, dim, dim)
            hessian_rank = hessian_rank[:, :supp, :supp]
            data[j]['rank_hessian'] = rank_h
            data[j]['svd_basis'] = hessian_rank
            hessian_full = torch.clone(hessian)
            hessian = hessian[:, :supp, :supp]
            data[j]['hessian'] = hessian_full

            P, BlockSizes = get_U(hessian_rank.to(torch.float64), epsilon1=1e-5, epsilon2=1e-5)
            data[j]['SBD_GT'] = BlockSizes
            data[j]['P'] = P
            print('BlockSizes alg: {} BlockSizes gt: {} '.format(BlockSizes,
                                                                 data[j]['groundtruth']['Blocs']))
            hessian_blocks = P @ hessian @ P.T
            hessian_rank_blocks = P @ hessian_rank @ P.T
            #print(hessian_rank_blocks.dtype, hessian_rank_blocks.abs().mean(dim=0))
            b_ad = 0
            b_ad_inner = 0
            for b in BlockSizes:
                if b > 1 and b <= 5:
                    data[j]['M']['gt']['Man_Opt'][b_ad_inner] = {}
                    data[j]['R']['gt']['Man_Opt'][b_ad_inner] = {}
                    data[j]['loss']['gt']['Man_Opt'][b_ad_inner] = {}
                    data[j]['time']['gt']['Man_Opt'][b_ad_inner] = {}

                    data[j]['M']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                    data[j]['R']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                    data[j]['loss']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                    data[j]['time']['gt']['Man_Opt_GS'][b_ad_inner] = {}

                    data[j]['M']['gt']['Grid_search'][b_ad_inner] = {}
                    data[j]['R']['gt']['Grid_search'][b_ad_inner] = {}
                    data[j]['loss']['gt']['Grid_search'][b_ad_inner] = {}
                    data[j]['time']['gt']['Grid_search'][b_ad_inner] = {}
                    hessian_rank_b = hessian_rank_blocks[:, b_ad:b_ad + b, b_ad:b_ad + b]
                    hessian_b = hessian_blocks[:, b_ad:b_ad + b, b_ad:b_ad + b]
                    batch_h = batches[b][h_size]
                    if h_size == 1 / 2 or run_man_opt:
                        result_man_opt = run_Man_Opt(
                            hessian_rank_b, v[b, b], optimizer_method=Method.Manifold_Opt,
                            h=h_size, batch_h=batch_h, N_epochs=N_epochs, dtype=dtype)
                        Bs_man_opt, losses_man_opt, times_man_opt = result_man_opt

                        data[j]['M']['gt']['Man_Opt'][b_ad_inner]['la'] = (
                                Bs_man_opt[0] @ hessian_b @ Bs_man_opt[0].T).abs().mean(dim=0)
                        data[j]['M']['gt']['Man_Opt'][b_ad_inner]['re'] = (
                                Bs_man_opt[1] @ hessian_b @ Bs_man_opt[1].T).abs().mean(dim=0)

                        data[j]['R']['gt']['Man_Opt'][b_ad_inner]['la'] = Bs_man_opt[0]
                        data[j]['R']['gt']['Man_Opt'][b_ad_inner]['re'] = Bs_man_opt[1]

                        data[j]['loss']['gt']['Man_Opt'][b_ad_inner]['la'] = losses_man_opt[0]
                        data[j]['loss']['gt']['Man_Opt'][b_ad_inner]['re'] = losses_man_opt[1]

                        data[j]['time']['gt']['Man_Opt'][b_ad_inner]['la'] = times_man_opt[0]
                        data[j]['time']['gt']['Man_Opt'][b_ad_inner]['re'] = times_man_opt[1]

                    result_man_opt_gs, result_grid_search = run_Man_Opt(
                        hessian_rank_b, v[b, b], optimizer_method=Method.Manifold_Opt_GS,
                        h=h_size, batch_h=batch_h, N_epochs=N_epochs, dtype=dtype)
                    Bs_man_opt_gs, losses_man_opt_gs, times_man_opt_gs = result_man_opt_gs

                    data[j]['M']['gt']['Grid_search'][b_ad_inner] = (
                            result_grid_search[0] @ hessian_b @ result_grid_search[0].T).abs().mean(
                        dim=0)
                    data[j]['M']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = (
                                Bs_man_opt_gs[0] @ hessian_b @ Bs_man_opt_gs[0].T).abs().mean(
                        dim=0)
                    data[j]['M']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = (
                                Bs_man_opt_gs[1] @ hessian_b @ Bs_man_opt_gs[1].T).abs().mean(
                        dim=0)

                    data[j]['R']['gt']['Grid_search'][b_ad_inner] = result_grid_search[0]
                    data[j]['R']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = Bs_man_opt_gs[0]
                    data[j]['R']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = Bs_man_opt_gs[1]

                    data[j]['loss']['gt']['Grid_search'][b_ad_inner] = result_grid_search[1]
                    data[j]['loss']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = losses_man_opt_gs[0]
                    data[j]['loss']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = losses_man_opt_gs[1]

                    data[j]['time']['gt']['Grid_search'][b_ad_inner] = result_grid_search[2]
                    data[j]['time']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = times_man_opt_gs[0] + \
                                                                            result_grid_search[2]
                    data[j]['time']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = times_man_opt_gs[1] + \
                                                                            result_grid_search[2]

                    b_ad_inner += 1
                b_ad += b
                suff_= '_cov_{}'.format(cov) if cov is not None else ''
                with open('{}/function_Man_opt_grid_search{}{}_h_{}_N_{}.json'.format(args.output_folder, N_run, suff_, args.h_size,
                                                                                  N_test_c), 'w') as convert_file:
                    json.dump(data, convert_file, cls=NumpyEncoder)









