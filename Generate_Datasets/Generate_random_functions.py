import torch
import numpy as np
import copy
from Utils.Generation_utils import ran_p, generate_cop,  generate_block_components
from Utils.Function_utils import compute_hessian_orig_2d, possible_sub_functions
from Utils.Evaluation_utils import NumpyEncoder
import math
import json
import sys
from os.path import dirname, abspath
if '/homes/math/ba/trafo_nova/' not in sys.path:
    sys.path.append('/homes/math/ba/trafo_nova/')
if '/homes/numerik/fatimaba/store/Github/trafo_nova/' not in sys.path:
    sys.path.append('/homes/numerik/fatimaba/store/Github/trafo_nova/')
if '//' not in sys.path:
    sys.path.append('//')

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def generate_components(input_data, v, cops, blocks,
                        max_subfunction_entries=2, probs=None,
                        poss_=None, start_k=1):
    '''

    :param K:
    :param dimension:
    :param max_block_size:
    :return:
    '''
    #np.random.seed(10)
    #torch.manual_seed(10)
    #if probs is None:
    #    probs = [.1, .1, .1, .1, .1, .5]
    dimension = input_data.shape[1]
    assert max_subfunction_entries <= dimension
    # build input output arrays, input x are drawn random normal
    num_samples = input_data.shape[0]
    out_f = torch.zeros(num_samples)
    # sum the output over all sub-functions
    t_num_sub_function_entries = []
    t_sub_function_indices = []
    parameters = []
    K=len(cops)
    y = torch.matmul(v, input_data.T).T
    for k in range(K):
        num_sub_function_entries = len(cops[k])
        poss = np.random.choice(range(len(possible_sub_functions)),
                                size=num_sub_function_entries) if poss_ is None else np.random.choice(poss_, size=num_sub_function_entries)
        sub_function_indices = poss
        # print(entry_indices, sub_function_indices, len(possible_sub_functions))
        t_sub_function_indices.append(sub_function_indices)
        t_num_sub_function_entries.append(num_sub_function_entries)
        # the sub-functions are multiplied, i.e. f_i(x_1, x_2, x_3) = f_k1(x_1) * f_k2(x_2) * f_k3(x_2)
        f_part = torch.ones(num_samples)
        ks = []
        for p in range(num_sub_function_entries):
            # get a random extra parameter for the sub function
            parameter = np.random.randint(0, 3) + start_k
            f_part *= possible_sub_functions[sub_function_indices[p]](y[:, cops[k][p]], parameter)
            ks += [parameter]
        parameters.append(ks)
        out_f += f_part
    ground_truth = {}
    ground_truth['n'] = dimension
    ground_truth['max_block_size'] = max([len(block) for block in blocks
                                          ]) if len(blocks)>0 else 2
    ground_truth['K'] = len(t_num_sub_function_entries)
    ground_truth['blocs'] = blocks
    ground_truth['J'] = cops
    ground_truth['num_samples'] = int(1e2)
    ground_truth['t_sub_function_indices'] = [u.tolist() for u in t_sub_function_indices]
    ground_truth['t_num_sub_function_entries'] = t_num_sub_function_entries
    ground_truth['parameters'] = parameters
    return out_f, ground_truth

if __name__ == '__main__':
    gen_block = True
    tmp1 = {
            'Grid_search': {},
            'Man_Opt': {
                'la': {},
                'rgd': {}
            },
            'Man_Opt_GS': {
                'la': {},
                'rgd': {}
            }
        }
    tmp = {'clean': copy.deepcopy(tmp1), 'noisy': copy.deepcopy(tmp1)}
    Output = {}
    N_run = 10
    output_folder = dirname(dirname(abspath(__file__)))+'/Test_Cases_Man_Opt_GS'
    #os.path.dirname(os.getcwd())+'/Anova_AE/Test_Cases_Man_Opt_GS'
    for j in range(N_run):
        Output[j] = {}
        dim = np.random.randint(4, 9) if gen_block else np.random.randint(5, 8)
        #N_hessian = 100 * dim
        #N_gradient = 100 * dim
        num_samples = int(1e2) * dim
        Output[j]['d'] = dim

        Output[j]['n'] = num_samples
        #Output[j]['n_g'] = N_gradient
        #Output[j]['n_h'] = N_hessian
        if gen_block:
            probs = np.array([.25, .37, .38])
            max_block_size = np.random.choice(range(3, 5), p=probs)
            K = math.floor(dim / max_block_size)
            # K = np.random.choice([K_, math.floor(K_ / 2) + 1], p=[0.5, 0.5])
            print("\n dim {} max_block_size {} K {} \n".format(dim, max_block_size, K))
            Output[j]['max_block_size'] = max_block_size
            Output[j]['K'] = K
            cop, blocks = generate_block_components(d=dim, K=K, max_block_size=max_block_size,
                                                    probs=np.array(probs[:max_block_size -1]) / np.sum(np.array(
                                                        probs[:max_block_size -1])), add_diag=False)
        else:
            s = np.random.randint(1, int(dim * (dim - 1) / 2 + 1))
            print('s= ', s)
            cop = generate_cop(d=dim, s=s)
            blocks = []
        R = ran_p(cop, d=dim)

        R = torch.as_tensor(R)
        input_data = (2 * torch.rand(num_samples, dim, dtype=torch.float64) - 1)
        target_val, ground_truth = generate_components(input_data, R, cop, blocks, poss_=[1, 2], start_k=2)
        Output[j]['x_test'] = input_data
        ground_truth['coeff'] = (20 - 5)*torch.rand(len(ground_truth['J'])) + 5
        Output[j]['time'] = copy.deepcopy(tmp)
        Output[j]['loss'] = copy.deepcopy(tmp)
        Output[j]['hessian_U'] = copy.deepcopy(tmp)
        Output[j]['J'] = copy.deepcopy(tmp)
        ground_truth['R'] = R
        hessianF_orig = lambda x_val: compute_hessian_orig_2d(
            x_val, ground_truth, return_coupling=True)
        hessian_o, J_ = hessianF_orig(x_val=input_data)
        copy_J_ = copy.deepcopy(J_)
        J = copy.deepcopy(ground_truth['J'])
        for u in J_:
            if u not in J and u[0] != u[1]:
                copy_J_.remove(u)
        #remain = [u for u in U_ if u not in U]
        print([u for u in copy_J_ if u not in J])
        ground_truth['J_true'] = copy_J_

        Num_non_zero_elements = []
        if gen_block:
            for bloc in ground_truth['blocs']:
                hessian_b_o = hessian_o[:, bloc, :]
                hessian_b_o = hessian_b_o[:, :, bloc].abs().mean(dim=0)
                hessian_b_o[hessian_b_o != torch.clamp(hessian_b_o, 1e-6)] = 0
                Num_non_zero_elements.append('b_size: {} num_nzero: {}'.format(len(bloc), len(hessian_b_o.nonzero())))
            print(Num_non_zero_elements)
            ground_truth['non_zero_block'] = Num_non_zero_elements
        Output[j]['groundtruth'] = ground_truth
        ground_truth['R'] = R
    print('\nsaving file.....')
    with open('{}/Test_functions_N_{}.json'.format(output_folder, N_run), 'w') as convert_file:
        json.dump(Output, convert_file, cls=NumpyEncoder)