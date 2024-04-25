'''
Running the  procedure for the matrix sets from section 5.1
with tilte: Manifold optimization on $SO(d)$ for jointly sparsifying a set of symmetric matrices
'''

import argparse
import copy
from os.path import dirname, abspath
import torch


from Utils.Evaluation_utils import NumpyEncoder
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    #device = torch.device('cuda')
    gdtype = torch.float64
else:
    #device = torch.device('cpu')
    gdtype = torch.float64

from Libs.Grid_Search import *

parser = argparse.ArgumentParser()

parser.add_argument('--output_folder', default=dirname(dirname(abspath(__file__))
                                                       ) + '/Output_algorithms/Random_matrices/Manifold_optimization',
                    type=str, help='Path of the Output folder')#'/work/ba/output'
parser.add_argument('--start_N_run', default=0, type=int,
                    help='Number of runs')
parser.add_argument('--N_run', default=100, type=int,
                    help='Number of runs')
parser.add_argument('--N_epochs', default=int(5e4)+1, type=int,
                    help='Number of iterations')
parser.add_argument('--suffix_file', default=100, type=int,
                    help='Differentiate file for each run')
parser.add_argument('--h_size', default=1, type=float,
                    help='Inner step_size Grid_seach')
parser.add_argument('--dim', default=2, type=int,
                    help='Input dimension')
parser.add_argument('-run_man_opt', action='store_true',
                    help='Var to Run manifold optimization with random init')
parser.add_argument('--print_mode', action='store_true',
                    help='print losses of the manifold optimization')
parser.add_argument('--opt_method', default='both', type=str,
                        help = 'Manifold optimization method: 1.RiemannianSGD, 2.LandingSGD, default:both methods')
parser.add_argument('--learning_rate', default=5e-4, type=int,
                        help='learning rate for the manifold optimization method')

# $ conda activate /work/ba/anova
args = parser.parse_args()
h_size = args.h_size
start_N_run = args.start_N_run
N_run = args.N_run
N_epochs = args.N_epochs
d = args.dim
run_man_opt = args.run_man_opt
print_mode = args.print_mode
opt_method = args.opt_method
learning_rate = args.learning_rate
in_dir_mo = dirname(dirname(abspath(__file__))) + '/Dataset'
in_dir = args.output_folder + '/Output_algorithms/Random_matrices/Grid_search'
batches = {
    2: {1: math.pi,
        1 / 2: math.pi,
        1 / 4: math.pi,
        1 / 8: math.pi,
        1/10: math.pi},
    3: {1: math.pi / 2,
        1 / 2: math.pi / 2,
        1 / 4: math.pi / 2,
        1 / 8: math.pi / 2,
        1/10: math.pi/2},
    4: {1: math.pi / 2,
        1 / 2: math.pi / 2,
        1 / 4: math.pi/2,
        1 / 8: math.pi/2,
        1/10: 1.0},
    5: {1: 2.36}
}
batch_h = batches[d][h_size]
if run_man_opt:
    with open('{}/Hessians_dim_{}_N_100.json'.format(in_dir_mo, d)) as convert_file:
        datas = copy.deepcopy(json.load(convert_file))
else:
    assert (h_size == 1 or h_size == 0.5 or h_size == 0.25 or h_size == 0.125 or h_size == 0.1)
    fname = '{}/Hessian_GS_{}_h_{}_bh_{:.2f}_dim_{}_gs.json'.format(
        in_dir, args.suffix_file, h_size, batch_h, d)
    with open(fname) as convert_file:
        datas = copy.deepcopy(json.load(convert_file))
for j in range(start_N_run, N_run):
    j = str(j)
    data = datas[j]
    print(data['J'])
    print('\nranK: ', data['rank']['clean'], 'rank_noisy: ', data['rank']['noisy'])
    data['h_size'] = h_size
    data['batch_h'] = batch_h
    hessian_rank = torch.as_tensor(data['svd_basis']['clean'], dtype=gdtype)
    hessian_rank_noise = torch.as_tensor(data['svd_basis']['noisy'], dtype=gdtype)
    hessian = torch.as_tensor(data['hessian']['clean'], dtype=gdtype)
    hessian_noise = torch.as_tensor(data['hessian']['noisy'], dtype=gdtype)
    R = torch.as_tensor(data['R'], dtype=gdtype)
    key_method = 'Man_Opt_RI' if run_man_opt else 'Man_Opt_GS'
    if run_man_opt:
        results_clean = run_Man_Opt(
            data, optimizer_method=Method.Manifold_Opt,
            h=h_size, batch_h=batch_h, N_epochs=N_epochs,
            print_mode=print_mode)
        Us_clean, losses_clean, _ = results_clean

        results_noisy = run_Man_Opt(
            data, optimizer_method=Method.Manifold_Opt,
            h=h_size, batch_h=batch_h, N_epochs=N_epochs,
            print_mode=print_mode, noisy_data=True, opt_method=opt_method,
            learning_rate=learning_rate)
        Us_noisy, losses_noisy, _ = results_noisy
    else:
        print('\n MAN_OPT_GS REIN')
        results_clean, _ = run_Man_Opt(
            data, optimizer_method=Method.Manifold_Opt_GS,
            h=h_size, batch_h=batch_h, N_epochs=N_epochs, print_mode=print_mode,
            opt_method=opt_method, learning_rate=learning_rate)
        Us_clean, losses_clean, _ = results_clean
        results_noisy, _ = run_Man_Opt(
            data, optimizer_method=Method.Manifold_Opt_GS,
            h=h_size, batch_h=batch_h, N_epochs=N_epochs, print_mode=print_mode, noisy_data=True)
        Us_noisy, losses_noisy, _ = results_noisy

    data['U']['clean'][key_method]['la'] = Us_clean[0]
    data['U']['clean'][key_method]['rgd'] = Us_clean[1]

    data['U']['noise'][key_method]['la'] = Us_noisy[0]
    data['U']['noise'][key_method]['rgd'] = Us_noisy[1]

    data['loss']['clean'][key_method]['la'] = losses_clean[0]
    data['loss']['clean'][key_method]['rgd'] = losses_clean[1]

    data['loss']['noise'][key_method]['la'] = losses_noisy[0]
    data['loss']['noise'][key_method]['rgd'] = losses_noisy[1]

    if run_man_opt:
        with open('{}/Out_comp/Compare_Man_opt_grid_search_{}_dim_{}.json'.format(
                args.output_folder, args.suffix_file, d), 'w') as convert_file:
            json.dump(datas, convert_file, cls=NumpyEncoder)
    else:
        with open('{}/Out_comp/Compare_Man_opt_grid_search_{}_h_{}_bh_{:.2f}_dim_{}.json'.format(
                args.output_folder, args.suffix_file, args.h_size, batch_h, d), 'w') as convert_file:
            json.dump(datas, convert_file, cls=NumpyEncoder)