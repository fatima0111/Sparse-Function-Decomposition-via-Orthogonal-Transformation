'''
Running the Grid-search initialization procedure for the matrix sets from section 5.1
with tilte: Manifold optimization on $SO(d)$ for jointly sparsifying a set of symmetric matrices
'''

import argparse
import copy
from os.path import dirname, abspath
import torch


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda')
    gdtype = torch.float64
else:
    device = torch.device('cpu')
    gdtype = torch.float64

from Libs.Grid_Search import *
from Utils.Evaluation_utils import NumpyEncoder
from Utils.Generation_utils import batches_random_matrices
parser = argparse.ArgumentParser()

parser.add_argument('--output_folder', default='/work/ba/output',
                    type=str, help='Path of the Output folder')  #
#'/homes/numerik/fatimaba/store/Github/trafo_nova/Anova_AE'
parser.add_argument('--start_N_run', default=0, type=int,
                    help='Number of runs')
parser.add_argument('--N_run', default=100, type=int,
                    help='Number of runs')
parser.add_argument('--suffix_file', default=100, type=int,
                    help='Differentiate file for each run')
parser.add_argument('--h_size', default=-1, type=float,
                    help='Inner step_size Grid_seach')
parser.add_argument('--dim', default=2, type=int,
                    help='Input dimension')

# $ conda activate /work/ba/anova
args = parser.parse_args()
h_size = args.h_size
start_N_run = args.start_N_run
N_run = args.N_run
dim = args.dim
in_dir = dirname(dirname(abspath(__file__)))+'/Test_Cases_Man_Opt_GS'
out_dir = args.output_folder + '/Grid_Search_Output'
batches = batches_random_matrices
if h_size in batches[dim].keys():
    batch_h = batches[dim][h_size]
    filename = '{}/Hessians_dim_{}_N_100.json'.format(in_dir, dim)
    print('\n {}'.format(filename), h_size)
    with open(filename) as convert_file:
        datas = copy.deepcopy(json.load(convert_file))
        new_datas = {}
        for j in range(start_N_run, N_run):
            new_datas[str(j)] = datas[str(j)]
        run_grid_search(new_datas, h_size, batch_h)
        with open('{}/Hessian_GS_{}_h_{}_bh_{:.2f}_dim_{}_gs.json'.format(
                out_dir, args.suffix_file, args.h_size, batch_h, dim), 'w') as convert_file:
            json.dump(new_datas, convert_file, cls=NumpyEncoder)
elif h_size == -1:
    for ch_size in batches[dim].keys():
        batch_h = batches[dim][ch_size]
        filename = '{}/Hessians_dim_{}_N_100.json'.format(in_dir, dim)
        print('\n {}'.format(filename), ch_size)
        with open(filename) as convert_file:
            datas = copy.deepcopy(json.load(convert_file))
            new_datas = {}
            for j in range(start_N_run, N_run):
                new_datas[str(j)] = datas[str(j)]
            run_grid_search(new_datas, ch_size, batch_h)
            print(new_datas)
            with open('{}/Hessian_GS_{}_h_{}_bh_{:.2f}_dim_{}_gs.json'.format(out_dir, args.suffix_file, ch_size,
                                                                            batch_h, dim), 'w') as convert_file:
                json.dump(new_datas, convert_file, cls=NumpyEncoder)
