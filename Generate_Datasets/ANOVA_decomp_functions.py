import torch
import numpy as np

def func1():
    '''
    f(x_1, ..., x_7) = sin(2x_1)*x_7^3 + exp(
        -(x_1-1)^2)*(x_4+3) + cos(2*x_2)*sin(3*x_5) + x_6^2
    :return: f(x_1, ..., x_7)
    '''

    d = 7
    blocks = [[0, 3, 6], [1, 4]]
    J = [[0, 6], [0, 3], [1, 4]]
    max_block_size = 3
    t_sub_function_indices = [[1, 0], [4, 5], [2, 5]]
    t_num_sub_function_entries = [2, 2, 2]
    parameters = [[2, 3], [1, 1], [2, 3]]
    coeff = [7, 5, 10]
    torch.manual_seed(10)
    np.random.seed(10)
    R = torch.as_tensor(np.linalg.svd(np.random.rand(d, d))[0], dtype=torch.float64)
    ground_truth = {}
    ground_truth['d'] = d
    ground_truth['max_block_size'] = max_block_size
    ground_truth['K'] = len(t_num_sub_function_entries)
    ground_truth['blocs'] = blocks
    ground_truth['J'] = J
    ground_truth['num_samples'] = int(1e2)
    ground_truth['t_sub_function_indices'] = [u for u in t_sub_function_indices]
    ground_truth['t_num_sub_function_entries'] = t_num_sub_function_entries
    ground_truth['parameters'] = parameters
    ground_truth['coeff'] = coeff
    ground_truth['R'] = R
    return ground_truth

def func2():
    '''
        f(x_1, ..., x_7) = sin(2x_1)*x_7^3 + exp(
            -(x_1-1)^2)*(x_4+3) + cos(2*x_2)*sin(3*x_5) + x_6^2
        :return: f(x_1, ..., x_7)
    '''

    d = 7
    blocks = [[0, 1, 3, 6], [2, 4, 5]]
    J = [[0, 6], [0, 3], [1, 6], [2, 4], [4, 5]]
    max_block_size = 4
    t_sub_function_indices = [[0, 0], [4, 2], [1, 2], [2, 1], [0, 0]]
    t_num_sub_function_entries = [2, 2, 2, 2, 2]
    parameters = [[1, 3], [1, 3], [1, 1], [2, 3], [1, 1]]
    coeff = [10, 5, 8, 12, 6]
    torch.manual_seed(10)
    np.random.seed(10)
    R = torch.as_tensor(np.linalg.svd(np.random.rand(d, d))[0], dtype=torch.float64)
    ground_truth = {}
    ground_truth['d'] = d
    ground_truth['max_block_size'] = max_block_size
    ground_truth['K'] = len(t_num_sub_function_entries)
    ground_truth['blocs'] = blocks
    ground_truth['J'] = J
    ground_truth['num_samples'] = int(1e2)
    ground_truth['t_sub_function_indices'] = [u for u in t_sub_function_indices]
    ground_truth['t_num_sub_function_entries'] = t_num_sub_function_entries
    ground_truth['parameters'] = parameters
    ground_truth['coeff'] = coeff
    ground_truth['R'] = R
    return ground_truth
