import torch
import numpy as np
import copy
from Utils.Generation_utils import ran_p, generate_cop, get_rank_svd
from Utils.Evaluation_utils import NumpyEncoder
import json
from os.path import dirname, abspath


if __name__ == '__main__':
    tmp1 = {
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
    tmp = {'clean': copy.deepcopy(tmp1), 'noisy': copy.deepcopy(tmp1)}
    Data = {}
    d = 5
    N_run = 100
    symmetric_noise = False
    output_folder = dirname(dirname(abspath(__file__)))+'/Dataset'
    for j in range(N_run):
        N_train = 100 * d
        s = np.random.randint(1, int(d * (d - 1) / 2 + 1))
        J = generate_cop(d=d, s=s)
        out = ran_p(J, d=d, N=N_train)
        matr, R = torch.tensor(out[0]), torch.tensor(out[1])
        noise_matrix = torch.normal(0, 1e-3, size=matr.shape)
        noise_matrix = torch.mm(noise_matrix, noise_matrix.T) if symmetric_noise else noise_matrix
        matr_noisy = matr + noise_matrix
        vec_matr = matr.flatten(start_dim=1).T
        vec_matr_noise = matr_noisy.flatten(start_dim=1).T
        u1, d1, v1h = torch.svd(vec_matr)
        u2, d2, v2h = torch.svd(vec_matr_noise)
        rank = get_rank_svd(d1, u1.shape[0])
        rank_noise = get_rank_svd(d2, u1.shape[0])
        #print('\nranK: ', rank)
        matr_rank = u1.T[:rank].reshape(rank, d, d)
        matr_rank_noise = u2.T[:rank_noise].reshape(rank_noise, d, d)
        Data[j] = {}
        Data[j]['d'] = d
        Data[j]['J'] = J
        Data[j]['R'] = R
        Data[j]['hessian_rank'] = {'clean': rank, 'noisy': rank_noise}
        Data[j]['hessian_basis'] = {'clean': matr_rank, 'noisy': matr_rank_noise}
        Data[j]['hessian'] = {'clean': matr, 'noisy': matr_noisy}
        Data[j]['time'] = copy.deepcopy(tmp)
        Data[j]['loss'] = copy.deepcopy(tmp)
        Data[j]['U'] = copy.deepcopy(tmp)
    with open('{}/matrices_dim_{}_M_{}.json'.format(output_folder, d, N_run), 'w') as convert_file:
        json.dump(Data, convert_file, cls=NumpyEncoder)