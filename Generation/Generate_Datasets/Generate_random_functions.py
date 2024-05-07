import torch
import numpy as np
import copy
from Utils.Generation_utils import ran_p,  generate_bloc_components
from Utils.Function_utils import compute_hessian_orig_2d, possible_sub_functions
from Utils.Evaluation_utils import NumpyEncoder
import math
import json
from os.path import dirname, abspath

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def generate_components(J, blocs):
    '''
    Constructs the function examples from section 5.2
    :param K: Number of edges (also number of sumands)
    :param blocs: Connected components on the graph
    :return: dictionary with information about the function
    '''
    t_sub_function_indices = []
    parameters = []
    K = len(J)
    for k in range(K):
        sub_function_indices = np.random.choice(range(len(possible_sub_functions)), size=2)
        var1 = np.random.choice([5, int(sub_function_indices[0])], p=[.55, .45])
        var2 = np.random.choice([5, int(sub_function_indices[1])], p=[.55, .45])
        sub_function_indices[0] = var1
        sub_function_indices[1] = var2

        t_sub_function_indices.append(sub_function_indices)
        parameter = np.random.choice(range(1, 3), size=2)
        parameters.append(parameter.tolist())
    ground_truth = {}
    ground_truth['t_sub_function_indices'] = [u.tolist() for u in t_sub_function_indices]
    ground_truth['parameters'] = parameters
    ground_truth['K'] = K
    ground_truth['max_bloc_size'] = max([len(bloc) for bloc in blocs]) if len(blocs) > 0 else 2
    ground_truth['J'] = J
    ground_truth['blocs'] = blocs
    return ground_truth

if __name__ == '__main__':
    gen_bloc = True
    tmp = {
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
    Output = {}
    N_run = 30
    output_folder = dirname(dirname(abspath(__file__)))+'/Dataset'
    for j in range(N_run):
        Output[j] = {}
        d = np.random.randint(10, 16) if gen_bloc else np.random.randint(5, 8)
        N = int(1e2) * d
        probs = np.array([.25, .37, .38])
        max_bloc_size = np.random.choice(range(2, 5), p=probs)
        K = math.floor(d / max_bloc_size)
        print("\n dim {} max_bloc_size {} K {} \n".format(d, max_bloc_size, K))
        J, blocs = generate_bloc_components(d=d, K=K, max_bloc_size=max_bloc_size,
                                              probs=np.array(probs[:max_bloc_size - 1]) / np.sum(np.array(
                                                    probs[:max_bloc_size - 1])), add_diag=False)
        R = ran_p(J, d=d)
        R = torch.as_tensor(R)
        input_data = (2 * torch.rand(N, d, dtype=torch.float64) - 1)
        ground_truth = generate_components(J, blocs)
        Output[j]['x_test'] = input_data
        ground_truth['coeff'] = (20 - 5)*torch.rand(len(ground_truth['J'])) + 5
        Output[j]['loss'] = copy.deepcopy(tmp)
        Output[j]['U'] = copy.deepcopy(tmp)
        ground_truth['R'] = R
        ground_truth['N'] = N
        ground_truth['d'] = d
        hessianF_orig = lambda x_val: compute_hessian_orig_2d(
            x_val, ground_truth, return_coupling=True)
        hessian_o, J_ = hessianF_orig(x_val=input_data)
        copy_J_ = copy.deepcopy(J_)
        J = copy.deepcopy(ground_truth['J'])
        for u in J_:
            if u not in J and u[0] != u[1]:
                copy_J_.remove(u)
        print([u for u in copy_J_ if u not in J])
        ground_truth['J_true'] = copy_J_
        Num_non_zero_elements = []
        if gen_bloc:
            for bloc in ground_truth['blocs']:
                hessian_b_o = hessian_o[:, bloc, :]
                hessian_b_o = hessian_b_o[:, :, bloc].abs().mean(dim=0)
                hessian_b_o[hessian_b_o != torch.clamp(hessian_b_o, 1e-6)] = 0
                Num_non_zero_elements.append('b_size: {} num_nzero: {}'.format(len(bloc), len(hessian_b_o.nonzero())))
            print(Num_non_zero_elements)
            ground_truth['non_zero_bloc'] = Num_non_zero_elements
            print('\n Num_non_zero_elements: ', Num_non_zero_elements)
        Output[j]['groundtruth'] = ground_truth
    print('\n saving file......')
    with open('{}/Test_functions_N_{}.json'.format(output_folder, N_run), 'w') as convert_file:
        json.dump(Output, convert_file, cls=NumpyEncoder)