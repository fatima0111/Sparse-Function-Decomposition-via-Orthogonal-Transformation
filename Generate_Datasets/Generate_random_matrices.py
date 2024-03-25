import torch
import numpy as np
import copy
from Utils.Generation_utils import ran_p, generate_cop, \
    get_rank_svd
from Utils.Evaluation_utils import NumpyEncoder
import json
import sys
from os.path import dirname, abspath
if '/homes/math/ba/trafo_nova/' not in sys.path:
    sys.path.append('/homes/math/ba/trafo_nova/')
if '/homes/numerik/fatimaba/store/Github/trafo_nova/' not in sys.path:
    sys.path.append('/homes/numerik/fatimaba/store/Github/trafo_nova/')
if '//' not in sys.path:
    sys.path.append('//')

if __name__ == '__main__':
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
    d = 5
    N_run = 100
    output_folder = dirname(abspath(__file__))+'/Test_Cases_Man_Opt_GS'
    #os.path.dirname(os.getcwd())+'/Anova_AE/Test_Cases_Man_Opt_GS'
    for j in range(N_run):
        N_train = 100 * d
        print(int(d * (d - 1) / 2))
        s = np.random.randint(1, int(d * (d - 1) / 2 + 1))
        print('s= ', s)
        cop = generate_cop(d=d, s=s)
        out = ran_p(cop, d=d, N=N_train)
        hessian, v = torch.tensor(out[0]), torch.tensor(out[1])
        hessian_noise = hessian + torch.normal(0, 1e-3, size=hessian.shape)
        vec_hessian = hessian.flatten(start_dim=1).T
        vec_hessian_noise = hessian_noise.flatten(start_dim=1).T
        u1, d1, v1h = torch.svd(vec_hessian)
        u2, d2, v2h = torch.svd(vec_hessian_noise)
        rank = get_rank_svd(d1, u1.shape[0])
        rank_noise = get_rank_svd(d2, u1.shape[0])
        print('\nranK: ', rank)
        hessian_rank = u1.T[:rank].reshape(rank, d, d)
        hessian_rank_noise = u2.T[:rank_noise].reshape(rank_noise, d, d)
        Output[j] = {}
        Output[j]['dim'] = d
        Output[j]['cop'] = cop
        Output[j]['v'] = v
        Output[j]['rank'] = rank
        Output[j]['rank_noise'] = rank_noise
        Output[j]['svd_basis'] = hessian_rank
        Output[j]['svd_basis_noisis'] = hessian_rank_noise
        Output[j]['hessian'] = hessian
        Output[j]['hessian_noise'] = hessian_noise
        Output[j]['time'] = copy.deepcopy(tmp)
        Output[j]['loss'] = copy.deepcopy(tmp)
        Output[j]['M'] = copy.deepcopy(tmp)
        Output[j]['R'] = copy.deepcopy(tmp)
    #print(Output)
    with open('{}/Hessians_dim_{}_N_{}.json'.format(output_folder, d, N_run), 'w') as convert_file:
        json.dump(Output, convert_file, cls=NumpyEncoder)