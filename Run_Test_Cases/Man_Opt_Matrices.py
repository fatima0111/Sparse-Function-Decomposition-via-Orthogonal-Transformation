import argparse
import copy
from os.path import dirname, abspath
import torch
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
sys.path.append('/homes/math/ba/trafo_nova/')
sys.path.append('//')
from Anova_AE.Libs.Grid_Search import *

parser = argparse.ArgumentParser()

parser.add_argument('--output_folder', default='/work/ba/output',
                    type=str, help='Path of the Output folder')#
parser.add_argument('--start_N_run', default=0, type=int,
                    help='Number of runs')
parser.add_argument('--N_run', default=100, type=int,
                    help='Number of runs')
parser.add_argument('--N_epochs', default=int(5e4)+1, type=int,
                    help='Number of epochs')
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

args = parser.parse_args()
#run_man_opt_gs = args.run_man_opt_gs
h_size = args.h_size

# $ conda activate /work/ba/anova
start_N_run = args.start_N_run
N_run = args.N_run  # 100
N_epochs = args.N_epochs
dim = args.dim
run_man_opt = args.run_man_opt
print_mode = args.print_mode
in_dir_mo = dirname(dirname(abspath(__file__)))+'/Test_Cases_Man_Opt_GS'
in_dir = args.output_folder + '/Grid_Search_Output'
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
    5: {1: 2.36,
        1 / 2: 1.5,
        1 / 4: 0.75}
}
batch_h = batches[dim][h_size]
if run_man_opt:
    with open('{}/Hessians_dim_{}_N_100.json'.format(in_dir_mo, dim)) as convert_file:
        datas = copy.deepcopy(json.load(convert_file))
else:
    assert (h_size == 1 or h_size == 0.5 or h_size == 0.25 or h_size == 0.125 or h_size == 0.1)
    fname = '{}/Hessian_GS_{}_h_{}_bh_{:.2f}_dim_{}_gs.json'.format(
        in_dir, args.suffix_file, h_size, batch_h, dim)
    with open(fname) as convert_file:
        datas = copy.deepcopy(json.load(convert_file))
for j in range(start_N_run, N_run):
    j = str(j)
    data = datas[j]
    print(data['cop'])
    print('\nranK: ', data['rank'], 'rank_noise: ', data['rank_noise'])
    data['h_size'] = h_size
    data['batch_h'] = batch_h
    hessian_rank = torch.as_tensor(data['svd_basis'], dtype=torch.float64)
    hessian_rank_noise = torch.as_tensor(data['svd_basis_noisis'], dtype=torch.float64)
    hessian = torch.as_tensor(data['hessian'], dtype=torch.float64)
    hessian_noise = torch.as_tensor(data['hessian_noise'], dtype=torch.float64)
    v = torch.as_tensor(data['v'], dtype=torch.float64)
    if run_man_opt:
        result_man_opt = run_Man_Opt(
            data, v, optimizer_method=Method.Manifold_Opt,
            h=h_size, batch_h=batch_h, N_epochs=N_epochs, print_mode=print_mode)
        Bs_man_opt, losses_man_opt, times_man_opt = result_man_opt

        result_man_opt_noise = run_Man_Opt(
            data, v, optimizer_method=Method.Manifold_Opt,
            h=h_size, batch_h=batch_h, N_epochs=N_epochs, print_mode=print_mode, noisy_data=True)
        Bs_man_opt_noise, losses_man_opt_noise, times_man_opt_noise = result_man_opt_noise

        data['M']['gt']['Man_Opt']['la'] = (Bs_man_opt[0] @ hessian @ Bs_man_opt[0].T).abs().mean(dim=0)
        data['M']['gt']['Man_Opt']['re'] = (Bs_man_opt[1] @ hessian @ Bs_man_opt[1].T).abs().mean(dim=0)

        data['M']['noise']['Man_Opt']['la'] = (Bs_man_opt_noise[0] @ hessian_noise @ Bs_man_opt_noise[0].T).abs().mean(
            dim=0)
        data['M']['noise']['Man_Opt']['re'] = (Bs_man_opt_noise[1] @ hessian_noise @ Bs_man_opt_noise[1].T).abs().mean(
            dim=0)
        data['R']['gt']['Man_Opt']['la'] = Bs_man_opt[0]
        data['R']['gt']['Man_Opt']['re'] = Bs_man_opt[1]

        data['R']['noise']['Man_Opt']['la'] = Bs_man_opt_noise[0]
        data['R']['noise']['Man_Opt']['re'] = Bs_man_opt_noise[1]

        data['loss']['gt']['Man_Opt']['la'] = losses_man_opt[0]
        data['loss']['gt']['Man_Opt']['re'] = losses_man_opt[1]

        data['loss']['noise']['Man_Opt']['la'] = losses_man_opt_noise[0]
        data['loss']['noise']['Man_Opt']['re'] = losses_man_opt_noise[1]
    else:
        print('\n MAN_OPT_GS REIN')
        result_man_opt_gs, result_grid_search = run_Man_Opt(
            data, v, optimizer_method=Method.Manifold_Opt_GS,
            h=h_size, batch_h=batch_h, N_epochs=N_epochs, print_mode=print_mode)
        Bs_man_opt_gs, losses_man_opt_gs, times_man_opt_gs = result_man_opt_gs
        result_man_opt_gs_noise, result_grid_search_noise = run_Man_Opt(
            data, v, optimizer_method=Method.Manifold_Opt_GS,
            h=h_size, batch_h=batch_h, N_epochs=N_epochs, print_mode=print_mode, noisy_data=True)
        Bs_man_opt_gs_noise, losses_man_opt_gs_noise, times_man_opt_gs_noise = result_man_opt_gs_noise
        #print('\n Overall time: ', t2-t1, time.time()-t2)

        #data['M']['gt']['Grid_search'] = (result_grid_search[0]@hessian@result_grid_search[0].T).abs().mean(dim=0)
        data['M']['gt']['Man_Opt_GS']['la'] = (Bs_man_opt_gs[0]@hessian@Bs_man_opt_gs[0].T).abs().mean(dim=0)
        data['M']['gt']['Man_Opt_GS']['re'] = (Bs_man_opt_gs[1]@hessian@Bs_man_opt_gs[1].T).abs().mean(dim=0)

        #data['M']['noise']['Grid_search'] = (
        #            result_grid_search_noise[0] @ hessian @ result_grid_search_noise[0].T).abs().mean(dim=0)
        data['M']['noise']['Man_Opt_GS']['la'] = (Bs_man_opt_gs_noise[0] @ hessian_noise @ Bs_man_opt_gs_noise[0].T).abs().mean(dim=0)
        data['M']['noise']['Man_Opt_GS']['re'] = (Bs_man_opt_gs_noise[1] @ hessian_noise @ Bs_man_opt_gs_noise[1].T).abs().mean(dim=0)

        #data['R']['gt']['Grid_search'] = result_grid_search[0]
        data['R']['gt']['Man_Opt_GS']['la'] = Bs_man_opt_gs[0]
        data['R']['gt']['Man_Opt_GS']['re'] = Bs_man_opt_gs[1]

        #data['R']['noise']['Grid_search'] = result_grid_search_noise[0]
        data['R']['noise']['Man_Opt_GS']['la'] = Bs_man_opt_gs_noise[0]
        data['R']['noise']['Man_Opt_GS']['re'] = Bs_man_opt_gs_noise[1]

        #data['loss']['gt']['Grid_search'] = result_grid_search[1]
        data['loss']['gt']['Man_Opt_GS']['la'] = losses_man_opt_gs[0]
        data['loss']['gt']['Man_Opt_GS']['re'] = losses_man_opt_gs[1]

        #data['loss']['noise']['Grid_search'] = result_grid_search_noise[1]
        data['loss']['noise']['Man_Opt_GS']['la'] = losses_man_opt_gs_noise[0]
        data['loss']['noise']['Man_Opt_GS']['re'] = losses_man_opt_gs_noise[1]


    #print(datas[j])
    if run_man_opt:
        with open('{}/Out_comp/Compare_Man_opt_grid_search_{}_dim_{}.json'.format(
                args.output_folder, args.suffix_file, dim), 'w') as convert_file:
            json.dump(datas, convert_file, cls=NumpyEncoder)
    else:
        with open('{}/Out_comp/Compare_Man_opt_grid_search_{}_h_{}_bh_{:.2f}_dim_{}.json'.format(
                args.output_folder, args.suffix_file, args.h_size, batch_h, dim), 'w') as convert_file:
            json.dump(datas, convert_file, cls=NumpyEncoder)