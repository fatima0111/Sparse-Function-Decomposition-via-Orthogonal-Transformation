import tntorch as tn
import operator
import itertools
import copy
import json
import torch
import numpy as np
from Utils.Function_utils import compute_function
from Utils.Evaluation_utils import get_total_Rot, Init_Method, NumpyEncoder
from Utils.Function_utils import compute_hessian_autograd, compute_gradient_autograd
from os.path import dirname, abspath
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda')
    gdtype = torch.float64
else:
    device = torch.device('cpu')
    gdtype = torch.float64
def findsubsets(d, n):
    '''
    :param d:
    :param n:
    :return:
    '''
    set_elem = set(itertools.combinations(set(range(d)), n))
    return [list(elem) for elem in list(set_elem)]


def inclusive_anova_term(anova, ind, coup,  comp, d=7, recons=False):
    '''

    :param anova:
    :param ind:
    :param coup:
    :param comp:
    :param d:
    :param recons:
    :return:
    '''
    N = d
    x0, x1, x2, x3, x4, x5, x6 = tn.symbols(N)
    xs = [x0, x1, x2, x3, x4, x5, x6]
    s_inter = len(ind)
    key = 'recons' if recons else 'gt'
    for j in range(s_inter, d):
        candidates = findsubsets(d, j)
        for upper in candidates:
            if set(ind) <= set(upper):
                if str(upper) in comp[key].keys():
                    ninf, n1 = comp[key][str(upper)]
                    coup[str(ind)][key][j]['inf-norm'].append(float(ninf))
                    coup[str(ind)][key][j]['1-norm'].append(float(n1))
                else: #Compute anova term
                    core = xs[upper[0]]
                    for i in range(1, len(upper)):
                        core = operator.and_(core, xs[upper[i]])
                    an_term = torch.as_tensor(tn.undo_anova_decomposition(
                        tn.mask(anova, tn.only(core))).numpy())
                    ninf = an_term.abs().max()
                    n1 = an_term.abs().mean()
                    comp[key][str(upper)] = [ninf, n1]
                    coup[str(ind)][key][j]['inf-norm'].append(float(ninf))
                    coup[str(ind)][key][j]['1-norm'].append(float(n1))
    return coup, comp


def get_vanishing_indices(ground_truth, grad, hess, U, max_inter=None, clamp=1e-4):
    '''

    :param ground_truth:
    :param grad:
    :param hess:
    :param U:
    :param max_inter:
    :param clamp:
    :return:
    '''
    d = ground_truth['d']
    if max_inter is None:
        max_inter = d
    grad_norm = U.T @ grad
    G_mo = grad_norm.abs().mean(dim=1)
    G_mo[G_mo != torch.clamp(G_mo, clamp)] = 0
    zero_norm_G_mo = (G_mo == 0).nonzero()
    s_cop = [[int(r)] for r in zero_norm_G_mo]
    sorted_couplings = [tuple(u) for u in list(map(sorted, s_cop))]
    first_order_indices = [list(u) for u in list(set(sorted_couplings))]
    first_order_indices.sort()
    hess_norm = U.T @ hess @ U
    H_mo = hess_norm.abs().mean(dim=0)
    H_mo[H_mo != torch.clamp(H_mo, clamp)] = 0
    zero_norm_H_mo = (H_mo == 0).nonzero()
    s_cop = [[int(r[0]), int(r[1])] for r in zero_norm_H_mo if int(r[0]) != int(r[1])]
    sorted_couplings = [tuple(u) for u in list(map(sorted, s_cop))]
    second_order_indices = [list(u) for u in list(set(sorted_couplings))]
    coopy = copy.deepcopy(second_order_indices)
    for ind_ in first_order_indices:
        for coupli in second_order_indices:
            if ind_[0] in coupli and coupli in coopy:
                coopy.remove(coupli)
    second_order_indices = coopy
    print(second_order_indices)
    #print('\n Hessian: ', H_mo, second_order_indices)
    two_interaction_set = findsubsets(d, 2)
    one_interaction_set = [[cpn] for cpn in range(d)]
    comp = {'gt': {}, 'recons': {}}
    coup1 = {}
    coup2 = {}
    coup1_max = {'gt': {}, 'recons': {}, 'grad': {}, 'labels': one_interaction_set}
    coup2_max = {'gt': {}, 'recons': {}, 'hessian': {}, 'labels': two_interaction_set}
    coup1_max['grad'] = {}
    coup2_max['hessian'] = {}
    coup1_max['grad']['inf-norm'] = []
    coup1_max['grad']['1-norm'] = []
    coup2_max['hessian'] = {}
    coup2_max['hessian']['inf-norm'] = []
    coup2_max['hessian']['1-norm'] = []
    coup1_max['gt']['inf-norm'] = {}
    coup1_max['gt']['1-norm'] = {}
    coup1_max['recons']['inf-norm'] = {}
    coup1_max['recons']['1-norm'] = {}
    coup2_max['gt']['inf-norm'] = {}
    coup2_max['gt']['1-norm'] = {}
    coup2_max['recons']['inf-norm'] = {}
    coup2_max['recons']['1-norm'] = {}
    coup1_max['gt']['inf-norm'] = []
    coup1_max['gt']['1-norm'] = []
    coup1_max['recons']['inf-norm'] = []
    coup1_max['recons']['1-norm'] = []
    coup2_max['gt']['inf-norm'] = []
    coup2_max['gt']['1-norm'] = []
    coup2_max['recons']['inf-norm'] = []
    coup2_max['recons']['1-norm'] = []
    axis = [torch.as_tensor(np.linspace(-1, 1, 12)).to(torch.device('cpu')) for j in range(d)]
    all = torch.meshgrid(*[axis[0] for j in range(d)])
    points = [all[j].flatten().to(torch.float32) for j in range(d)]
    points = torch.stack(points, dim=1).T
    R = torch.as_tensor(ground_truth['R'], dtype=torch.float32)
    points_ = U @ points
    f_gt = compute_function((R.T @ points).T, ground_truth).to(torch.device('cpu')).reshape(all[0].shape)
    f_recons = compute_function(points_.T, ground_truth).to(torch.device('cpu')).reshape(all[0].shape)

    t_gt = tn.Tensor(f_gt)
    t_recons = tn.Tensor(f_recons)
    anova_gt = tn.anova_decomposition(t_gt)
    anova_recons = tn.anova_decomposition(t_recons)
    if len(first_order_indices) != 0:
        for cp in one_interaction_set:#first_order_indices:
            coup1[str(cp)] = {'gt': {}, 'recons': {}, 'grad': [grad_norm.abs().max(dim=1)[0][cp[0]],
                                                                grad_norm.abs().mean(dim=1)[cp[0]]]}
            coup1_max['grad']['inf-norm'].append(grad_norm.abs().max(dim=1)[0][cp[0]])
            coup1_max['grad']['1-norm'].append(grad_norm.abs().mean(dim=1)[cp[0]])

            for j in range(1, max_inter):
                coup1[str(cp)]['gt'][j] = {'inf-norm': [], '1-norm':[]}
                coup1[str(cp)]['recons'][j] = {'inf-norm': [], '1-norm':[]}
            coup1, comp = inclusive_anova_term(anova_gt, cp, coup1, comp, d=7)
            coup1, comp = inclusive_anova_term(anova_recons, cp, coup1, comp, d=7, recons=True)
            coup1_max['recons']['inf-norm'].append(max([max(coup1[str(cp)]['recons'][j]['inf-norm']) for j in range(1, max_inter)]))
            coup1_max['recons']['1-norm'].append(max([max(coup1[str(cp)]['recons'][j]['1-norm']) for j in range(1, max_inter)]))
    print("First order couplings: ", coup1_max)
    if len(second_order_indices) > 0:
        for cp in two_interaction_set:
            coup2[str(cp)] = {'gt': {}, 'recons': {}, 'hessian': [hess_norm.abs().max(dim=0)[0][cp[0], cp[1]],
                                                                hess_norm.abs().mean(dim=0)[cp[0], cp[1]]]}
            coup2_max['hessian']['inf-norm'].append(hess_norm.abs().max(dim=0)[0][cp[0], cp[1]])
            coup2_max['hessian']['1-norm'].append(hess_norm.abs().mean(dim=0)[cp[0], cp[1]])

            for j in range(2, max_inter):
                coup2[str(cp)]['gt'][j] = {'inf-norm': [], '1-norm': []}
                coup2[str(cp)]['recons'][j] = {'inf-norm': [], '1-norm': []}
            coup2, comp = inclusive_anova_term(anova_gt, cp, coup2, comp, d=7)
            coup2, comp = inclusive_anova_term(anova_recons, cp, coup2, comp, d=7, recons=True)
            coup2_max['recons']['inf-norm'].append(max([max(coup2[str(cp)]['recons'][j]['inf-norm']) for j in range(2, max_inter)]))
            coup2_max['recons']['1-norm'].append(max([max(coup2[str(cp)]['recons'][j]['1-norm']) for j in range(2, max_inter)]))
    print("Second order couplings: ", coup2_max)
    return {'1': coup1_max, '2': coup2_max}


if __name__ == '__main__':
    in_dir = dirname(dirname(abspath(__file__))) + '/Output/Output_algorithms/ANOVA_sparse_functions'
    #out_dir = dirname(dirname(abspath(__file__))) + '/homes/numerik/fatimaba/store/Github/trafo_nova/Anova_AE/Output_files'
    covs = [None, 0.5]
    init_method = Init_Method.GS
    h_sizes = [1.0]
    for h_size in h_sizes:
        print("\n .................................h_size: ", h_size)
        for cov in covs:
            suff_ = '_cov_{}'.format(cov) if cov is not None else ''
            if init_method == Init_Method.GS:
                name = '{}/Test_ANOVA{}_h_size_{}.json'.format(in_dir, suff_, h_size)
            else:
                name = '{}/Test_ANOVA{}_MO_RI.json'.format(in_dir, suff_)
            with open(name) as convert_file:
                datas = copy.deepcopy(json.load(convert_file))
                for j in datas.keys():
                    ground_truth = datas[str(j)]['groundtruth']
                    #d = ground_truth['d']
                    U_la, U_rgd = get_total_Rot(datas, str(j), init_method=init_method)
                    hessian_supp = torch.as_tensor(datas['0']['hessian_basis'])
                    supp = hessian_supp.shape[1]
                    N = hessian_supp.shape[0]
                    x_test = torch.as_tensor(datas[str(j)]['x_test'])
                    ground_truth['R'] = torch.as_tensor(ground_truth['R'])
                    U = U_rgd.T
                    gradient = compute_gradient_autograd(x_test, ground_truth)
                    hessian = compute_hessian_autograd(x_test, ground_truth)
                    results_ = get_vanishing_indices(ground_truth, gradient, hessian, U)
                    #print(results_)
                    suff_ = '_cov_{}'.format(cov) if cov is not None else ''
                    if init_method == Init_Method.GS:
                        with open('{}/ANOVA_deriv{}_h_size_{}_{}.json'.format(in_dir, suff_, h_size, j),
                                  'w') as convert_file:
                            json.dump(results_, convert_file, cls=NumpyEncoder)
                    else:
                        with open('{}/ANOVA_deriv{}_MO_RI_{}.json'.format(in_dir, suff_, j), 'w') as convert_file:
                            json.dump(results_, convert_file, cls=NumpyEncoder)

