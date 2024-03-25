import torch
import numpy as np
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.autograd.functional import hessian
import itertools
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

possible_sub_functions = [lambda x, p: x ** p,
                          lambda x, p: torch.sin(x * p),
                          lambda x, p: torch.cos(x * p),
                          lambda x, p: (x ** 2 + p ** 2)**(1/3),
                          lambda x, p: torch.exp(-(x-p)**2),
                          lambda x, p: x + p
                          ]

first_order_derivate = [lambda x, p: p*(x**(p-1)),
                              lambda x, p: p*torch.cos(x*p),
                              lambda x, p: -p*torch.sin(x * p),
                              lambda x, p: (2*x)/(3*(x**2 +p**2)**(2/3)),
                              lambda x, p: -2*torch.exp(-(x-p)**2)*(x-p),
                              lambda x, p: torch.ones(x.shape[0])
                          ]
second_order_derivative = [lambda x, p: (p-1)*p*(x**(p-2)) if p-2>=0 else p*torch.zeros(x.shape[0]),
                           lambda x, p: -(p**2)*(torch.sin(x*p)),
                           lambda x, p: -(p**2)*torch.cos(x * p),
                           lambda x, p: -2*(x**2-3*p**2)/(9*(x**2 +p**2)**(5/3)),
                           lambda x, p: torch.exp(-(x-p)**2)*(4*(x**2)-8*p*x+4*(p**2)-2),
                           lambda x, p: torch.zeros(x.shape[0])

]


def compute_function(x, ground_truth):
    """
    For computing the i-th anova_function just set num_sub_functions=[i]
    :param input_data:
    :param num_sub_functions: list of involved anova functions can be range of N if N functions are involved
    :param num_sub_function_entries:
    :param sub_function_indices:
    :param entry_indices:
    :param parameters:
    :return:
    """
    if x.ndim == 1:
        x = x.unsqueeze(dim=0)
    num_sub_functions = ground_truth['K']
    R = ground_truth['v']
    entry_indices = ground_truth['U']
    parameters = ground_truth['parameters']
    num_sub_function_entries = ground_truth['t_num_sub_function_entries']
    sub_function_indices = ground_truth['t_sub_function_indices']
    input_data = torch.matmul(R, x.T).T
    # the sub-functions are multiplied, i.e. f_i(x_1, x_2, x_3) = f_k1(x_1) * f_k2(x_2) * f_k3(x_2)
    num_samples = input_data.shape[0]
    out_f = torch.zeros(num_samples)
    coeff = ground_truth['coeff'] if 'coeff' in ground_truth.keys() else torch.ones(num_sub_functions)
    if type(num_sub_functions) == int:
        num_sub_functions = list(range(num_sub_functions))
    for i in num_sub_functions:
        f_part = torch.ones(num_samples)
        for k in range(num_sub_function_entries[i]):
            parameter = parameters[i][k]
            f_part *= possible_sub_functions[sub_function_indices[i][k]](
                input_data[:, entry_indices[i][k]], parameter)
        out_f += coeff[i]*f_part
    return out_f


def proj_orth(Q):
    u, d, v_h = torch.linalg.svd(Q)
    Q = torch.matmul(u, v_h)
    return Q

def compute_hessian_autograd(x, ground_truth):
    '''

    :param x:
    :param ground_truth:
    :param model:
    :return:
    '''
    num_samples, data_dimension = x.shape
    from torch.autograd.functional import hessian
    def f(ground_truth):
        def get_random(x):
            return compute_function(x, ground_truth=ground_truth)
        return get_random
    hessian_ = torch.zeros((num_samples, data_dimension, data_dimension))
    for i in range(num_samples):
        x_i = x[i, :]
        x_i = x_i
        x_i.requires_grad = True
        val = hessian(f(ground_truth), x_i)
        hessian_[i, :, :] = val.squeeze()
    return hessian_

def compute_gradient_autograd(x, ground_truth):
    '''
    :param x:
    :param ground_truth:
    :param model:
    :return:
    '''
    def f(ground_truth):
        def get_random(x):
            return compute_function(x, ground_truth=ground_truth)
        return get_random
    gradient = torch.zeros_like(x.T)
    for i in range(x.shape[0]):
        x_i = x[i:i+1, :]
        x_i.required_grad = True
        gradient[:, i] = torch.autograd.functional.jacobian(f(ground_truth), x_i, create_graph=True)
    return gradient.detach()



def compute_gradient_orig_2d(x, ground_truth, n_A=None,
                             return_coupling=False):
    '''

    :param x:
    :param ground_truth:
    :param n_A:
    :param return_coupling:
    :return:
    '''
    n = x.shape[1]
    U = ground_truth['U']
    K = ground_truth['K']
    A = ground_truth['v'][n_A] if n_A is not None else ground_truth['v']
    y = torch.matmul(A, x.T).T
    gradient_f = torch.zeros([x.shape[1], x.shape[0]])
    t_sub_function_indices = ground_truth['t_sub_function_indices']
    parameters = ground_truth['parameters']
    active_variables = list(itertools.chain(*U))
    active_variables.sort()
    active_variables = list(set(active_variables))
    coeff = ground_truth['coeff'] if 'coeff' in ground_truth.keys() else torch.ones(K)
    for i in range(n):
        gradient_i = torch.zeros(x.shape[0])
        if i in active_variables:
            for k in range(K):
                ind_u = U.index(U[k])
                f_k = t_sub_function_indices[ind_u]
                p_k = parameters[ind_u]
                if i in U[k]:
                    ind_i = U[k].index(i)
                    if U[k][0] == U[k][1]:
                        gradient_i += coeff[k]*first_order_derivate[f_k[ind_i]](
                            y[:, i], p_k[ind_i])
                    elif U[k][0] != U[k][1]:
                        gradient_i += coeff[k]*first_order_derivate[f_k[ind_i]](
                            y[:, i], p_k[ind_i])*possible_sub_functions[
                            f_k[(ind_i+1) % 2]](
                            y[:, U[k][(ind_i+1) % 2]], p_k[(ind_i+1) % 2]
                        )
            gradient_f[i, :] = gradient_i
    if return_coupling:
        return gradient_f
    gradient = torch.matmul(A.T, gradient_f)
    return gradient


def compute_hessian_orig_2d(x, ground_truth, return_coupling=False,
                            n_A=None):
    '''

    :param x:
    :param ground_truth:
    :param return_coupling:
    :param n_A:
    :return:
    '''
    n = x.shape[1]
    U = ground_truth['U']
    K = ground_truth['K']
    A = ground_truth['v'][n_A] if n_A is not None else ground_truth['v']
    y = torch.matmul(A, x.T).T
    hessian_f = torch.zeros([x.shape[0], x.shape[1], x.shape[1]])
    t_sub_function_indices = ground_truth['t_sub_function_indices']
    parameters = ground_truth['parameters']
    active_variables = list(itertools.chain(*U))
    active_variables.sort()
    active_variables = list(set(active_variables))
    coeff = ground_truth['coeff'] if 'coeff' in ground_truth.keys() else torch.ones(K)
    for i in range(n):
        for j in range(n):
            hessian_ij = torch.zeros(x.shape[0])
            if i in active_variables and j in active_variables:
                for k in range(K):
                    ind_u = U.index(U[k])
                    if i in U[k] and j in U[k]:
                        ind_i = U[k].index(i)
                        ind_j = U[k].index(j)
                        f_k = t_sub_function_indices[ind_u]
                        p_k = parameters[ind_u]
                        if i == j and i in U[k]:
                            if U[k][0] == U[k][1]:
                                hessian_ij += coeff[k]*second_order_derivative[f_k[ind_i]](x[:, i], p_k[ind_i])
                            elif U[k][0] != U[k][1]:
                                hessian_ij += coeff[k]*second_order_derivative[f_k[ind_i]](
                                    y[:, i], p_k[ind_i])*possible_sub_functions[
                                    f_k[(ind_i+1) % 2]](
                                    y[:, U[k][(ind_i+1) % 2]], p_k[(ind_i+1) % 2]
                                )
                        elif i != j and i in U[k] and j in U[k]:
                            hessian_ij += coeff[k]*first_order_derivate[f_k[ind_i]](
                                y[:, i], p_k[ind_i])*first_order_derivate[f_k[ind_j]](
                                y[:, j], p_k[ind_j])
            hessian_f[:, i, j] = hessian_ij
    if return_coupling:
        hessian_orig = hessian_f.abs().mean(dim=0)
        hessian_orig[hessian_orig != torch.clamp(hessian_orig, 1e-4)] = 0
        C = torch.where(hessian_orig > 0)
        return hessian_f, [[C[0][i].item(), C[1][i].item()] for i in range(len(C[0]))]
    hessian = torch.matmul(torch.matmul(A.T, hessian_f), A)
    return hessian

def noise_function(sample, means_=None, cov=.5, type='0', dtype= torch.float64):
    '''

    :param sample:
    :param means:
    :param cov:
    :param type: 0=Gradient pdf, 1=Hessian pdf, else:pdf
    :return:
    '''
    d = sample.shape[1]
    cs = {0.5: 2000,
          1: 7500,
          1.5: 60000,
          2: 10000
}
    denom = 1 / cs[cov]
    cov_mat = cov * torch.eye(d)
    den = -1 / (2 * cov)
    if means_ is None:
        means_ = [-.5, 1.5]
    means_list = []
    for i in range(d):
        means_list.append(means_)
    means = list(itertools.product(*means_list))
    exp = torch.zeros(len(means), sample.shape[0])
    mean_v = torch.as_tensor(means)
    for ind_mean, mean in enumerate(means):
        mean = torch.as_tensor(mean)
        diff = sample - torch.as_tensor(mean)
        exp[ind_mean, :] = torch.exp(-((diff @ torch.linalg.inv(cov_mat)) * diff).sum(dim=1) / 2)
    if type == '0': #Gradient
        gradient = torch.zeros_like(sample.T,  dtype=dtype)
        for i in range(d):
            mean_i = mean_v[:, i]
            diff_i = mean_i - torch.tile(sample[:, i], (mean_i.shape[0], 1)).T
            gradient[i, :] = (den*diff_i.T*exp).sum(dim=0)
        return denom*gradient
    elif type == '1': #Hessian
        hessian = torch.zeros((sample.shape[0], d, d))
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    mean_i = mean_v[:, i]
                    diff_i = mean_i-torch.tile(sample[:, i], (mean_i.shape[0], 1)).T
                    hessian_i = (den*(den*diff_i.T**2 +1)*exp).sum(dim=0)
                else:
                    diff_i = mean_v[:, i] - torch.tile(sample[:, i], (mean_v[:, i].shape[0], 1)).T
                    diff_j = mean_v[:, j] - torch.tile(sample[:, j], (mean_v[:, j].shape[0], 1)).T
                    hessian_i = (den**2 * diff_i.T*diff_j.T*exp).sum(dim=0)
                hessian[:, i, j] = hessian[:, j, i] = hessian_i
        return hessian*denom
    else:
        return exp.sum(dim=0)*denom

