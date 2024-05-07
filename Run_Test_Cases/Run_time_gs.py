from os.path import dirname, abspath
import torch
from Utils.Evaluation_utils import NumpyEncoder
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

from Libs.Grid_Search import *
import json
import copy
import argparse
from Utils.Generation_utils import batches_random_matrices
parser = argparse.ArgumentParser()
parser.add_argument('--ind_j', default=0, type=int,
                    help='Which test case')
args = parser.parse_args()
ind_j = args.ind_j
N_run = 10
batches = batches_random_matrices
dims = [2, 3, 4, 5]

out_dir = dirname(dirname(abspath(__file__)))+'/Output_algorithms/Random_matrices/Time_complexity'
random_indices = [[59], [48], [17], [31], [50], [9], [27], [80], [5], [43]]
random_index = [random_indices[ind_j]]
in_dir = dirname(dirname(abspath(__file__)))+'/Dataset'

for dim in dims:
    filename = '{}/matrices_dim_{}_M_100.json'.format(in_dir, dim)
    print('\n Running 5-time random initialization method: ', filename)
    with open(filename) as convert_file:
        datas = copy.deepcopy(json.load(convert_file))
        all_data = {}
        for ind_ in random_index:
            j = str(ind_[0])
            data = datas[j]
            out = run_Man_Opt(data, optimizer_method=Method.Manifold_Opt, N_epochs=100)
            Us, losses, times_man_opt = out
            data['time']['clean']['Man_Opt_RI']['la'] = times_man_opt[0]
            data['time']['clean']['Man_Opt_RI']['rgd'] = times_man_opt[1]
            all_data[j] = data
        with open('{}/Time_Complexity_RI_{}_dim_{}_gs.json'.format(out_dir, N_run+ind_j, dim), 'w') as convert_file_mo:
            json.dump(all_data, convert_file_mo, cls=NumpyEncoder)
            print('saving: ', '{}/Time_Complexity_RI_{}_dim_{}_gs.json'.format(out_dir, N_run+ind_j, dim))

    print('\n Running Grid-Search:')
    h_sizes = [1.0, 0.5, 0.25, 0.125, 0.1] if dim < 5 else [1.0]
    for h_size in h_sizes:
        suffix_file = len(random_index)
        all_data = {}
        batch_h = batches[dim][h_size]
        for inner_j in random_index:
            filename = '{}/matrices_dim_{}_M_100.json'.format(in_dir, dim)
            with open(filename) as convert_file:
                datas = copy.deepcopy(json.load(convert_file))
                new_datas = {}
                for j in inner_j:
                    new_datas[str(j)] = datas[str(j)]
                run_grid_search(new_datas, h_size, batch_h)
                for j in inner_j:
                    all_data[str(j)] = new_datas[str(j)]
        with open('{}/Time_Complexity_GS_{}_h_{}_bh_{:.2f}_dim_{}_gs.json'.format(
                out_dir, N_run+ind_j, h_size, batch_h, dim), 'w') as convert_file1:
            json.dump(all_data, convert_file1, cls=NumpyEncoder)
            print('saving: ', '{}/Time_Complexity_GS_{}_h_{}_bh_{:.2f}_dim_{}_gs.json'.format(
                out_dir, N_run+ind_j, h_size, batch_h, dim))
