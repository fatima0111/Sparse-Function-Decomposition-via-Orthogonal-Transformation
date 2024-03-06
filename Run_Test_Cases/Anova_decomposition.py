import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tntorch as tn

from Anova_AE.Libs.Utils import  random_function, compute_hessian_orig_2d, \
    compute_gradient_orig_2d, noise_function, compute_hessian_auto, compute_gradient_autograd
from Anova_AE.Libs.Grid_Search import *
from Anova_AE.Libs.roots import get_rank_svd, get_U
from Anova_AE.Evaluate_Test_Cases.Evaluation_functions import get_total_Rot

def func1(u, v, w, x, y, z):
    return 0*u + 0*v + 0*w + np.sin(y * x)+z+x**2

def func2(return_f):
    '''
    f(x_1, ..., x_7) = sin(2x_1)*x_7^3 + exp(
        -(x_1-1)^2)*(x_4+3) + cos(2*x_2)*sin(3*x_5) + x_6^2
    :return: f(x_1, ..., x_7)
    '''

    dim = 7
    blocks = [[0, 3, 6], [1, 4]]
    cops = [[0, 6], [0, 3], [1, 4]]
    max_components = 3
    t_sub_function_indices = [[1, 0], [4, 5], [2, 5]]
    t_num_sub_function_entries = [2, 2, 2]
    parameters = [[2, 3], [1, 1], [2, 3]]
    coeff = [7, 5, 10]
    torch.manual_seed(10)
    np.random.seed(10)
    v = torch.as_tensor(np.linalg.svd(np.random.rand(dim, dim))[0], dtype=torch.float64)

    ground_truth = {}
    ground_truth['n'] = dim
    ground_truth['max_components'] = max_components
    ground_truth['K'] = len(t_num_sub_function_entries)
    ground_truth['Blocs'] = blocks  # [block.tolist() for block in blocks]
    ground_truth['U'] = cops
    ground_truth['num_samples'] = int(1e2)
    ground_truth['t_sub_function_indices'] = [u for u in t_sub_function_indices]
    ground_truth['t_num_sub_function_entries'] = t_num_sub_function_entries
    ground_truth['parameters'] = parameters
    ground_truth['coeff'] = coeff

    ground_truth['v'] = v
    return ground_truth
def func2_old():
    '''
    f(x_1, ..., x_7) = 7*sin(2x_1)*x_7^3 + 5*exp(
        -(x_1-1)^2)*(x_4+1) + 10*cos(2*x_2)*sin(3*x_5)
    :return: f(x_1, ..., x_7)
    '''

    dim = 7
    blocks = [[0, 3, 6], [1, 4]]
    cops = [[0, 6], [0, 3], [1, 4]]
    max_components = 3
    t_sub_function_indices = [[1, 0], [4, 5], [2, 5]]
    t_num_sub_function_entries = [2, 2, 2]
    parameters = [[2, 3], [1, 1], [2, 3]]
    coeff = [7, 5, 10]
    torch.manual_seed(10)
    np.seed(10)
    v = torch.as_tensor(np.linalg.svd(np.random.rand(dim, dim))[0], dtype=torch.float64)

    ground_truth = {}
    ground_truth['n'] = dim
    ground_truth['max_components'] = max_components
    ground_truth['K'] = len(t_num_sub_function_entries)
    ground_truth['Blocs'] = blocks  # [block.tolist() for block in blocks]
    ground_truth['U'] = cops
    ground_truth['num_samples'] = int(1e2)
    ground_truth['t_sub_function_indices'] = [u for u in t_sub_function_indices]
    ground_truth['t_num_sub_function_entries'] = t_num_sub_function_entries
    ground_truth['parameters'] = parameters
    ground_truth['coeff'] = coeff
    torch.manual_seed(10)
    ground_truth['v'] = v
    return ground_truth
def func3():
    '''
        f(x_1, ..., x_7) = sin(2x_1)*x_7^3 + exp(
            -(x_1-1)^2)*(x_4+3) + cos(2*x_2)*sin(3*x_5) + x_6^2
        :return: f(x_1, ..., x_7)
    '''

    dim = 7
    blocks = [[0, 1, 3, 6], [2, 4, 5]]
    cops = [[0, 6], [0, 3], [1, 6], [2, 4], [4, 5]]
    max_components = 4
    t_sub_function_indices = [[0, 0], [4, 2], [1, 2], [2, 1], [0, 0]]
    t_num_sub_function_entries = [2, 2, 2, 2, 2]
    parameters = [[1, 3], [1, 3], [1, 1], [2, 3], [1, 1]]
    coeff = [10, 5, 8, 12, 6]
    torch.manual_seed(10)
    np.random.seed(10)
    v = torch.as_tensor(np.linalg.svd(np.random.rand(dim, dim))[0], dtype=torch.float64)
    ground_truth = {}
    ground_truth['n'] = dim
    ground_truth['max_components'] = max_components
    ground_truth['K'] = len(t_num_sub_function_entries)
    ground_truth['Blocs'] = blocks  # [block.tolist() for block in blocks]
    ground_truth['U'] = cops
    ground_truth['num_samples'] = int(1e2)
    ground_truth['t_sub_function_indices'] = [u for u in t_sub_function_indices]
    ground_truth['t_num_sub_function_entries'] = t_num_sub_function_entries
    ground_truth['parameters'] = parameters
    ground_truth['coeff'] = coeff
    ground_truth['v'] = v
    return ground_truth

def set_gt(ground_truths):
    import copy
    from Anova_AE.Libs.Utils import compute_gradient_orig_2d, compute_hessian_orig_2d
    datas = {}
    for ind_g, ground_truth in enumerate(ground_truths):
        dim = ground_truth['n']
        num_samples = int(1e2) * dim
        torch.manual_seed(10)
        input_data = (2 * torch.rand(num_samples, dim, dtype=torch.float64) - 1)
        hessianF_orig = lambda x_val: compute_hessian_orig_2d(
            x_val, ground_truth, return_coupling=True)
        hessian_o, U_ = hessianF_orig(x_val=input_data)
        copy_U_ = copy.deepcopy(U_)
        U = copy.deepcopy(ground_truth['U'])
        for u in U_:
            if u not in U and u[0] != u[1]:
                copy_U_.remove(u)
        # remain = [u for u in U_ if u not in U]
        print([u for u in copy_U_ if u not in U])
        ground_truth['U_true'] = copy_U_


        Num_non_zero_elements = []
        for bloc in ground_truth['Blocs']:
            hessian_b_o = hessian_o[:, bloc, :]
            hessian_b_o = hessian_b_o[:, :, bloc].abs().mean(dim=0)
            hessian_b_o[hessian_b_o != torch.clamp(hessian_b_o, 1e-6)] = 0
            Num_non_zero_elements.append('b_size: {} num_nzero: {}'.format(len(bloc), len(hessian_b_o.nonzero())))
        print(Num_non_zero_elements)
        ground_truth['non_zero_block'] = Num_non_zero_elements

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
        #Output['max_components'] = max_components
        Output['K'] = len(ground_truth['U'])
        Output['dim'] = dim
        Output['n'] = num_samples
        Output['x_test'] = input_data
        Output['time'] = copy.deepcopy(tmp)
        Output['loss'] = copy.deepcopy(tmp)
        Output['M'] = copy.deepcopy(tmp)
        Output['R'] = copy.deepcopy(tmp)
        Output['groundtruth'] = ground_truth
        datas[str(ind_g)] = Output
    return datas

def train_model(datas, output_folder, N_epochs = int(5e4), run_man_opt=True, h_size=None):
    if run_man_opt:
        Compute_rotatation(datas, output_folder, N_epochs=N_epochs, run_man_opt=True, cov=None)
        Compute_rotatation(datas, output_folder, N_epochs=N_epochs,
                           run_man_opt=True, cov=0.5)
    else:
        h_sizes = [h_size] if h_size is not None else [1, 0.5, 0.25, 0.125]
        for h_size in h_sizes:
            Compute_rotatation(datas, output_folder, N_epochs=N_epochs, run_man_opt=run_man_opt,
                               cov=None, h_size=h_size)

            Compute_rotatation(datas, output_folder, N_epochs=N_epochs,
                               run_man_opt=run_man_opt, cov=0.5, h_size=h_size)
def Compute_rotatation(datas, output_folder, h_size=None,
               N_epochs = int(5e4), run_man_opt=True, cov=None):

    batches = {
        2: {3: math.pi,
            1: math.pi,
            1 / 2: math.pi,
            1 / 4: math.pi,
            1 / 8: math.pi},
        3: {3: math.pi,
            1: math.pi / 2,
            1 / 2: math.pi / 2,
            1 / 4: math.pi / 2,
            1 / 8: math.pi / 2},
        4: {3: math.pi,
            1: math.pi / 2,
            1 / 2: math.pi / 2,
            1 / 4: math.pi / 2,
            1 / 8: 1.0}
    }
    eps1 = 3 if cov is not None else 3
    eps2 = 2 if cov is not None else 3
    dtype = torch.float64
    for j in range(2):
        data = datas[str(j)]
        ground_truth = data['groundtruth']
        dim = ground_truth['n']
        v = ground_truth['v']
        x_test = data['x_test']
        #y_test = random_function(x_test, ground_truth)
        gradF_orig = compute_gradient_orig_2d(x_test, ground_truth)
        gradient = gradF_orig
        hessianF_orig = compute_hessian_orig_2d(x_test.clone(), ground_truth)
        if cov is not None:
            gradient_noise = noise_function(x_test.clone(), cov=cov, type='0', dtype=dtype)
            hessian_noise = noise_function(x_test.clone(), cov=cov, type='1', dtype=dtype)
            gradient =+ gradient_noise
            hessianF_orig += hessian_noise

        u1, d1, v1 = torch.svd(gradient)
        data['grad_U'] = u1
        supp = get_rank_svd(d1, u1.shape[0], eps=eps1)
        data['support'] = supp
        hessian_ = torch.matmul(torch.matmul(u1.T.to(device), hessianF_orig), u1.to(device))
        hessian = hessian_

        vec_hessian = hessian_.flatten(start_dim=1).T
        u2, d2, v2 = torch.svd(vec_hessian)
        rank_h = get_rank_svd(d2, u2.shape[0], eps=eps2)
        hessian_rank = u2.T[:rank_h].reshape(rank_h, dim, dim)
        hessian_rank = hessian_rank[:, :supp, :supp]
        data['rank_hessian'] = rank_h
        data['svd_basis'] = hessian_rank
        hessian_full = torch.clone(hessian)
        hessian = hessian[:, :supp, :supp]
        data['hessian'] = hessian_full

        P, BlockSizes = get_U(hessian_rank.to(torch.float64), epsilon1=1e-5, epsilon2=1e-5)
        data['SBD_GT'] = BlockSizes
        data['P'] = P
        print('BlockSizes alg: {} BlockSizes gt: {} '.format(BlockSizes,
                                                             data['groundtruth']['Blocs']))
        hessian_blocks = P @ hessian @ P.T
        hessian_rank_blocks = P @ hessian_rank @ P.T
        # print(hessian_rank_blocks.dtype, hessian_rank_blocks.abs().mean(dim=0))
        b_ad = 0
        b_ad_inner = 0
        for b in BlockSizes:
            if b > 1 and b <= 5:
                data['M']['gt']['Man_Opt'][b_ad_inner] = {}
                data['R']['gt']['Man_Opt'][b_ad_inner] = {}
                data['loss']['gt']['Man_Opt'][b_ad_inner] = {}
                data['time']['gt']['Man_Opt'][b_ad_inner] = {}

                data['M']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                data['R']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                data['loss']['gt']['Man_Opt_GS'][b_ad_inner] = {}
                data['time']['gt']['Man_Opt_GS'][b_ad_inner] = {}

                data['M']['gt']['Grid_search'][b_ad_inner] = {}
                data['R']['gt']['Grid_search'][b_ad_inner] = {}
                data['loss']['gt']['Grid_search'][b_ad_inner] = {}
                data['time']['gt']['Grid_search'][b_ad_inner] = {}
                hessian_rank_b = hessian_rank_blocks[:, b_ad:b_ad + b, b_ad:b_ad + b]
                hessian_b = hessian_blocks[:, b_ad:b_ad + b, b_ad:b_ad + b]

                if run_man_opt:
                    result_man_opt = run_MO_Block(
                        hessian_rank_b, v[b, b], optimizer_method=Method.Manifold_Opt, N_epochs=N_epochs)
                    Bs_man_opt, losses_man_opt, times_man_opt = result_man_opt

                    data['M']['gt']['Man_Opt'][b_ad_inner]['la'] = (
                            Bs_man_opt[0] @ hessian_b @ Bs_man_opt[0].T).abs().mean(dim=0)
                    data['M']['gt']['Man_Opt'][b_ad_inner]['re'] = (
                            Bs_man_opt[1] @ hessian_b @ Bs_man_opt[1].T).abs().mean(dim=0)

                    data['R']['gt']['Man_Opt'][b_ad_inner]['la'] = Bs_man_opt[0]
                    data['R']['gt']['Man_Opt'][b_ad_inner]['re'] = Bs_man_opt[1]

                    data['loss']['gt']['Man_Opt'][b_ad_inner]['la'] = losses_man_opt[0]
                    data['loss']['gt']['Man_Opt'][b_ad_inner]['re'] = losses_man_opt[1]

                    data['time']['gt']['Man_Opt'][b_ad_inner]['la'] = times_man_opt[0]
                    data['time']['gt']['Man_Opt'][b_ad_inner]['re'] = times_man_opt[1]
                else:
                    batch_h = batches[b][h_size]
                    result_man_opt_gs, result_grid_search, time_grid_search = run_MO_Block(
                        hessian_rank_b, v[b, b], optimizer_method=Method.Manifold_Opt_GS,
                        h=h_size, batch_h=batch_h, N_epochs=N_epochs)
                    Bs_man_opt_gs, losses_man_opt_gs, times_man_opt_gs = result_man_opt_gs

                    data['M']['gt']['Grid_search'][b_ad_inner] = (
                            result_grid_search[0] @ hessian_b @ result_grid_search[0].T).abs().mean(dim=0)
                    data['M']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = (
                            Bs_man_opt_gs[0] @ hessian_b @ Bs_man_opt_gs[0].T).abs().mean(dim=0)
                    data['M']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = (
                            Bs_man_opt_gs[1] @ hessian_b @ Bs_man_opt_gs[1].T).abs().mean(dim=0)

                    data['R']['gt']['Grid_search'][b_ad_inner] = result_grid_search[0]
                    data['R']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = Bs_man_opt_gs[0]
                    data['R']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = Bs_man_opt_gs[1]

                    data['loss']['gt']['Grid_search'][b_ad_inner] = result_grid_search[1]
                    data['loss']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = losses_man_opt_gs[0]
                    data['loss']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = losses_man_opt_gs[1]

                    data['time']['gt']['Grid_search'][b_ad_inner] = time_grid_search
                    data['time']['gt']['Man_Opt_GS'][b_ad_inner]['la'] = times_man_opt_gs[0]
                    data['time']['gt']['Man_Opt_GS'][b_ad_inner]['re'] = times_man_opt_gs[1]
                b_ad_inner += 1
            b_ad += b
            suff_ = '_cov_{}'.format(cov) if cov is not None else ''
            if not run_man_opt:
                with open('{}/Test_anova{}_h_size_{}.json'.format(output_folder, suff_, h_size), 'w') as convert_file:
                    json.dump(datas, convert_file, cls=NumpyEncoder)
            else:
                with open('{}/Test_anova{}_mo.json'.format(output_folder, suff_), 'w') as convert_file:
                    json.dump(datas, convert_file, cls=NumpyEncoder)
    return datas

def findsubsets(dim, n):
    '''
    :param dim:
    :param n:
    :return:
    '''
    set_elem = set(itertools.combinations(set(range(dim)), n))
    return [list(elem) for elem in list(set_elem)]


def ndm(*args):
    return [x[(None,)*i+(slice(None),)+(None,)*(len(args)-i-1)] for i, x in enumerate(args)]
def test_Dummy_anova():
    #torch.set_default_tensor_type('torch.cpu')
    N = 6
    xaxis = torch.as_tensor(np.linspace(-1, 1, 25))
    yaxis = torch.as_tensor(np.linspace(-1, 1, 25))
    zaxis = torch.as_tensor(np.linspace(-1, 1, 25))
    uaxis = torch.as_tensor(np.linspace(-1, 1, 25))
    vaxis = torch.as_tensor(np.linspace(-1, 1, 25))
    waxis = torch.as_tensor(np.linspace(-1, 1, 25))

    u2, v2, w2, x2, y2, z2 = ndm(uaxis, vaxis, waxis, xaxis, yaxis, zaxis)
    result3 = func1(u2, v2, w2, x2, y2, z2).to(torch.float32)
    print(result3.shape, u2.shape, x2.shape)
    t = tn.Tensor(result3)
    x, y, z, u, v, w = tn.symbols(N)
    anova = tn.anova_decomposition(t)
    fo = tn.undo_anova_decomposition(tn.mask(anova, tn.none(N))).numpy()
    fx = tn.undo_anova_decomposition(tn.mask(anova, tn.only(x))).numpy()
    fy = tn.undo_anova_decomposition(tn.mask(anova, tn.only(y))).numpy()
    fz = tn.undo_anova_decomposition(tn.mask(anova, tn.only(z))).numpy()
    fu = tn.undo_anova_decomposition(tn.mask(anova, tn.only(u))).numpy()
    fxy = tn.undo_anova_decomposition(tn.mask(anova, tn.only(x & y))).numpy()
    fxz = tn.undo_anova_decomposition(tn.mask(anova, tn.only(x & z))).numpy()
    fyz = tn.undo_anova_decomposition(tn.mask(anova, tn.only(y & z))).numpy()
    fxu = tn.undo_anova_decomposition(tn.mask(anova, tn.only(x & u))).numpy()
    fxyz = tn.undo_anova_decomposition(tn.mask(anova, tn.only(x & y & z))).numpy()
    f_terms = [fo, fx, fy, fz, fu, fxy, fxz, fyz, fxu, fxyz]
    labels = ['fo', 'fx', 'fy', 'fz', 'fu', 'fxy', 'fxz', 'fyz', 'fxu', 'fxyz']
    for i, an_term in enumerate(f_terms):
        an_term = torch.as_tensor(an_term)
        print(labels[i], ': ', an_term.abs().max(),
              an_term.abs().mean(), (an_term ** 2).abs().sqrt())

def inclusive_anova_term(anova, ind, coup,  comp, d=7, recons=False, clamp=1e-4):
    import operator
    N = dim
    x0, x1, x2, x3, x4, x5, x6 = tn.symbols(N)
    xs = [x0, x1, x2, x3, x4, x5, x6]
    s_inter = len(ind)#+1
    key = 'recons' if recons else 'gt'
    for j in range(s_inter, d):
        candidates = findsubsets(d, j)
        for upper in candidates:
            if set(ind) <= set(upper):
                if str(upper) in comp[key].keys():
                    ninf, n1 = comp[key][str(upper)]
                    #if ninf <= clamp:
                    coup[str(ind)][key][j]['inf-norm'].append(float(ninf))# += 1
                    #if n1 <= clamp:
                    coup[str(ind)][key][j]['1-norm'].append(float(n1))# += 1
                else: #Compute anova term
                    core = xs[upper[0]]
                    for i in range(1, len(upper)):
                        core = operator.and_(core, xs[upper[i]])
                    an_term = torch.as_tensor(tn.undo_anova_decomposition(
                        tn.mask(anova, tn.only(core))).numpy())
                    ninf = an_term.abs().max()
                    n1 = an_term.abs().mean()
                    comp[key][str(upper)] = [ninf, n1]
                    coup[str(ind)][key][j]['inf-norm'].append(float(ninf))#[0] += 1
                    coup[str(ind)][key][j]['1-norm'].append(float(n1)) #+= 1
    return coup, comp

def get_vanishing_indices(ground_truth, grad, hess, R, max_inter=None, clamp=1e-4):
    d = ground_truth['n']
    if max_inter is None:
        max_inter = d
    grad_norm = R.T@ grad
    G_mo = grad_norm.abs().mean(dim=1)
    G_mo[G_mo != torch.clamp(G_mo, clamp)] = 0
    zero_norm_G_mo = (G_mo == 0).nonzero()
    s_cop = [[int(r)] for r in zero_norm_G_mo]
    sorted_couplings = [tuple(u) for u in list(map(sorted, s_cop))]
    first_order_indices = [list(u) for u in list(set(sorted_couplings))]
    first_order_indices.sort()
    #print('\n gradient: ', G_mo, first_order_indices)
    hess_norm = R.T @ hess @ R
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
    comp = {'gt':{}, 'recons':{}}
    coup1 = {}
    coup2 = {}
    coup1_max = {'gt':{}, 'recons':{}, 'grad':{}, 'labels':one_interaction_set}
    coup2_max = {'gt':{}, 'recons':{}, 'hessian':{}, 'labels':two_interaction_set}

    coup1_max['grad'] = {}
    coup2_max['hessian'] = {}
    coup1_max['grad']['inf-norm'] = []
    coup1_max['grad']['1-norm'] = []
    coup2_max['hessian'] = {}
    coup2_max['hessian']['inf-norm'] = []
    coup2_max['hessian']['1-norm'] = []
    #print(ground_truth)

    axis = [torch.as_tensor(np.linspace(-1, 1, 12)).to(torch.device('cpu')) for j in range(d)]
    all = torch.meshgrid(*[axis[0] for j in range(d)])
    points = [all[j].flatten().to(torch.float32) for j in range(d)]
    points = torch.stack(points, dim=1).T
    A = torch.as_tensor(ground_truth['v'], dtype=torch.float32)
    points_ = R @ points
    f_gt = random_function((A.T@points).T, ground_truth).to(torch.device('cpu')).reshape(all[0].shape)
    f_recons = random_function(points_.T, ground_truth).to(torch.device('cpu')).reshape(all[0].shape)

    t_gt = tn.Tensor(f_gt)
    t_recons = tn.Tensor(f_recons)

    anova_gt = tn.anova_decomposition(t_gt)
    anova_recons = tn.anova_decomposition(t_recons)
    coup1_max['gt']['inf-norm'] = {}
    coup1_max['gt']['1-norm'] = {}
    coup1_max['recons']['inf-norm'] = {}
    coup1_max['recons']['1-norm'] = {}
    coup2_max['gt']['inf-norm'] = {}
    coup2_max['gt']['1-norm'] = {}
    coup2_max['recons']['inf-norm'] = {}
    coup2_max['recons']['1-norm'] = {}
    #for p in range(1, max_inter):
    coup1_max['gt']['inf-norm'] = []
    coup1_max['gt']['1-norm'] = []
    coup1_max['recons']['inf-norm'] = []
    coup1_max['recons']['1-norm'] = []

    coup2_max['gt']['inf-norm'] = []
    coup2_max['gt']['1-norm'] = []
    coup2_max['recons']['inf-norm'] = []
    coup2_max['recons']['1-norm'] = []

    if len(first_order_indices) != 0:
        for cp in one_interaction_set:#first_order_indices:
            coup1[str(cp)] = {'gt':{}, 'recons':{}, 'grad': [grad_norm.abs().max(dim=1)[0][cp[0]],
                                                            grad_norm.abs().mean(dim=1)[cp[0]]]}
            coup1_max['grad']['inf-norm'].append(grad_norm.abs().max(dim=1)[0][cp[0]])
            coup1_max['grad']['1-norm'].append(grad_norm.abs().mean(dim=1)[cp[0]])

            for j in range(1, max_inter):
                coup1[str(cp)]['gt'][j] = {'inf-norm': [], '1-norm':[]}
                coup1[str(cp)]['recons'][j] = {'inf-norm': [], '1-norm':[]}
            coup1, comp = inclusive_anova_term(anova_gt, cp, coup1, comp, d=7, clamp=clamp)
            coup1, comp = inclusive_anova_term(anova_recons, cp, coup1, comp, d=7, clamp=clamp, recons=True)
            coup1_max['recons']['inf-norm'].append(max([max(coup1[str(cp)]['recons'][j]['inf-norm']) for j in range(1, max_inter)]))
            coup1_max['recons']['1-norm'].append(max([max(coup1[str(cp)]['recons'][j]['1-norm']) for j in range(1, max_inter)]))
    print("First order couplings: ", coup1_max)
    if len(second_order_indices)>0:

        for cp in two_interaction_set:#second_order_indices:
            coup2[str(cp)] = {'gt':{}, 'recons':{}, 'hessian': [hess_norm.abs().max(dim=0)[0][cp[0], cp[1]],
                                                                hess_norm.abs().mean(dim=0)[cp[0], cp[1]]]}
            coup2_max['hessian']['inf-norm'].append(hess_norm.abs().max(dim=0)[0][cp[0], cp[1]])
            coup2_max['hessian']['1-norm'].append(hess_norm.abs().mean(dim=0)[cp[0], cp[1]])

            for j in range(2, max_inter):
                coup2[str(cp)]['gt'][j] = {'inf-norm': [], '1-norm':[]}
                coup2[str(cp)]['recons'][j] = {'inf-norm': [], '1-norm':[]}
            coup2, comp = inclusive_anova_term(anova_gt, cp, coup2, comp, d=7, clamp=clamp)
            coup2, comp = inclusive_anova_term(anova_recons, cp, coup2, comp, d=7, clamp=clamp, recons=True)
            coup2_max['recons']['inf-norm'].append(max([max(coup2[str(cp)]['recons'][j]['inf-norm']) for j in range(2, max_inter)]))
            coup2_max['recons']['1-norm'].append(max([max(coup2[str(cp)]['recons'][j]['1-norm']) for j in range(2, max_inter)]))
    print("Second order couplings: ", coup2_max)
    return {'1':coup1_max, '2':coup2_max}
def n_order_anova_terms(anova, dim, order=None, interactions = None, compute_min=False, clamp=5e-5):
    '''

    :param t:
    :param dim:
    :param order:
    :param clamp:
    :return:
    '''
    import operator
    #operator.and_(True, False)
    anova_terms_0 = []
    anova_terms_1 = []
    #anova_terms_2 = []
    ind_anova_terms_0 = []
    ind_anova_terms_1 = []
    #ind_anova_terms_2 = []
    N=dim
    x0, x1, x2, x3, x4, x5, x6 = tn.symbols(N)
    if order is not None and interactions is None:
        interactions = findsubsets(dim, order)
    xs = [x0, x1, x2, x3, x4, x5, x6]
    for ind_inter, inter in enumerate(interactions):
        core = xs[inter[0]]
        for i in range(1, len(inter)):
            core = operator.and_(core, xs[inter[i]])
        an_term = torch.as_tensor(tn.undo_anova_decomposition(
        tn.mask(anova, tn.only(core))).numpy())
        #print(inter, ':', an_term.abs().max(), an_term.abs().mean(), ((an_term**2).mean()).sqrt())
        if compute_min:
            clamp = an_term.abs().max()
        if an_term.abs().max() <= clamp:
            anova_terms_0.append(an_term.abs().max())
            ind_anova_terms_0.append(interactions)
        if an_term.abs().mean() <= clamp:
            anova_terms_1.append(an_term.abs().mean())
            ind_anova_terms_1.append(interactions)
        #if ((an_term**2).mean()).sqrt() < clamp:
        #    anova_terms_2.append(((an_term**2).mean()).sqrt())
        #    ind_anova_terms_2.append(interactions)
    return [anova_terms_0, anova_terms_1], [ind_anova_terms_0, ind_anova_terms_1]

def extract_upper_diag(A):
    A = A.numpy()
    i = A.shape[0]
    k, l = np.triu_indices(i, 1)
    return torch.tensor(A[k, l])
def compare_anova_der(ground_truth, R, max_components=2, clamp=1e-4):
    #print(ground_truth)
    d = ground_truth['n']
    axis = [torch.as_tensor(np.linspace(-1, 1, 12)).to(torch.device('cpu')) for j in range(d)]
    all = torch.meshgrid(*[axis[0] for j in range(d)])
    points = [all[j].flatten().to(torch.float32) for j in range(d)]
    points = torch.stack(points, dim=1).T
    A = torch.as_tensor(ground_truth['v'], dtype=torch.float32)
    #print('A @ Rot: ', A @ R)
    points_ = R @ points
    f_gt = random_function((A.T@points).T, ground_truth).to(torch.device('cpu')).reshape(all[0].shape)
    f_disturb = random_function(points.T, ground_truth).to(torch.device('cpu')).reshape(all[0].shape)
    f_recons = random_function((R @ points).T, ground_truth).to(torch.device('cpu')).reshape(all[0].shape)

    t_gt = tn.Tensor(f_gt)
    t_disturb = tn.Tensor(f_disturb)
    t_recons = tn.Tensor(f_recons)

    anova_gt = tn.anova_decomposition(t_gt)
    anova_disturb = tn.anova_decomposition(t_disturb)
    anova_recons = tn.anova_decomposition(t_recons)
    results = {'anova_terms': {'gt': {}, 'disturb':{}, 'recons': {}}, 'partial_der': {}}
    for order in range(1, max_components+1):
        anovas_gt, indices_gt = n_order_anova_terms(anova_gt, d, order, clamp=5e-5)
        anovas_disturb, indices_disturb = n_order_anova_terms(anova_disturb, d, order,compute_min=True, clamp=5e-5)
        anovas_recons, indices_recons = n_order_anova_terms(anova_recons, d, order, clamp=5e-5)
        results['anova_terms']['gt'][order] = [len(anovas_gt[0]), len(anovas_gt[1])]
        results['anova_terms']['disturb'][order] = [min(anovas_disturb[0]), min(anovas_disturb[1])]
        results['anova_terms']['recons'][order] = [len(anovas_recons[0]), len(anovas_recons[1])]
        if order <= 2:
            if order == 1:
                gradient = compute_gradient_autograd(points[:, :500].T, ground_truth)
                G_mo1 = (R.T@gradient).abs().mean(dim=1)
                G_mo1[G_mo1 != torch.clamp(G_mo1, clamp)] = 0
                zero_norm_G_mo1 = (G_mo1 == 0).nonzero()

                G_mo0 = (R.T @ gradient).abs().max(dim=1)[0]
                G_mo0[G_mo0 != torch.clamp(G_mo0, clamp)] = 0
                zero_norm_G_mo0 = (G_mo0 == 0).nonzero()
                results['partial_der']['grad'] = [len(zero_norm_G_mo0), len(zero_norm_G_mo1)]
            else:
                hessian = compute_hessian_auto(points[:, :500].T, ground_truth)
                H_mo1 = (R.T@hessian@R).abs().mean(dim=0)
                H_mo1[H_mo1 != torch.clamp(H_mo1, clamp)] = 0
                zero_norm_H_mo1 = (extract_upper_diag(H_mo1) == 0).nonzero()

                H_mo0 = (R.T @ hessian @ R).abs().max(dim=0)[0]
                H_mo0[H_mo0 != torch.clamp(H_mo0, clamp)] = 0
                zero_norm_H_mo0 = (extract_upper_diag(H_mo0) == 0).nonzero()
                results['partial_der']['hess'] = [len(zero_norm_H_mo1), len(zero_norm_H_mo1)]
                #print('\nhessian: ', len(zero_norm_H_mo))
        print("\n max_components: ", max_components, 'order: ', order, results)
    return results


def anova_decomposition(ground_truth, indices1, indices2, Rot, v, U,  hess, dim=7,):
    N=dim
    axis = [torch.as_tensor(np.linspace(-1, 1, 12)).to(torch.device('cpu')) for d in range(dim)]
    #a, b, c, d, e, f, g = ndm(*axis)
    #print(a.get_device(), g.get_device())
    all = torch.meshgrid(*[axis[0] for d in range(dim)])
    print(all[0].flatten().dtype)
    points = [all[i].flatten().to(torch.float32) for i in range(dim)]
    points = torch.stack(points, dim=1).T
    points_ = v@Rot@points
    print(points_.shape, points[:, :100].shape)
    gt2 = datas[str(j)]['groundtruth']
    gt2['v'] = torch.tensor(gt2['v']).clone()@Rot
    hessian2 = compute_hessian_auto(points[:, :100].T, gt2)
    print('\n hessian rot: ', hessian2.abs().mean(dim=0))
    result3 = random_function(points_.T, ground_truth).to(torch.device('cpu')).reshape(all[0].shape)
    print(result3.shape)

    #print(all[0].shape, points_.shape)
    #print(result3.shape, result3.dtype, result3.get_device())
    t = tn.Tensor(result3)
    x0, x1, x2, x3, x4, x5, x6 = tn.symbols(N)

    xs = [x0, x1, x2, x3, x4, x5, x6]
    anova = tn.anova_decomposition(t)
    f1 = []
    f1_labels = []
    fo = tn.undo_anova_decomposition(tn.mask(anova, tn.none(N))).numpy()
    for i1 in indices1:
        a_term = torch.as_tensor(tn.undo_anova_decomposition(
            tn.mask(anova, tn.only(xs[i1]))).numpy())
        f1.append(a_term)
        f1_labels.append('f_{}'.format(str(i1)))
        print('f_{}'.format(str(i1)), ':',  a_term.abs().max(),
              a_term.abs().mean(), (a_term ** 2).abs().mean().sqrt())
    f2 = []
    f2_labels = []
    #ind1, ind2 = indices2
    for ik in indices2: #zip(ind1, ind2):
        i1, i2 = ik
        a_term = torch.as_tensor(tn.undo_anova_decomposition(
            tn.mask(anova, tn.only(xs[i1]&xs[i2]))).numpy())
        f2.append(a_term)
        f2_labels.append('f_{}'.format(str(i1)+str(i2)))
        clamp = 1e-6
        H_mo = (Rot.T@U@hess@U.T@Rot).abs().mean(dim=0)
        H_mo[H_mo!= torch.clamp(H_mo, clamp)] = 0
        #zero_norm_H_mo = (H_mo == 0).nonzero()
        print('f_{}'.format(str(i1)+str(i2)),
              a_term.abs().mean(), H_mo[i1, i2])
    a_term = torch.as_tensor(tn.undo_anova_decomposition(
        tn.mask(anova, tn.only(xs[0] & xs[2] & xs[3]))).numpy())
    print('f_{}'.format(str(0) + str(2)+str(3)),
          a_term.abs().mean())
    a_term = torch.as_tensor(tn.undo_anova_decomposition(
        tn.mask(anova, tn.only(xs[0] & xs[2] & xs[3]&xs[4]&xs[5]))).numpy())
    print('f_{}'.format(str(0) + str(2) + str(3)+ str(4)+ str(5)),
          a_term.abs().mean())


def plot_histogram(data, grad_=False, j=str(0), clamp=1e-4, cov=None):
    import matplotlib.pyplot as plt
    import numpy as np
    print(data)
    end = 7 if j == str(0) else 8
    key = '1' if grad_ else '2'
    norm = '\infty'#
    if len(data[key]['labels'])>0:
        #name= 'Gradient' if grad_ else 'Hessian'
        name_partial = "Gradient" if grad_ else "Hessian" #"partial_inf" if norm=='\infty' else "partial_1"
        #name_anova = "anova_inf" if norm=='\infty' else "anova_1"
        name1 = 'grad' if grad_ else 'hessian'
        norm_key = '1-norm' if norm == '1' else 'inf-norm'
        p_indices = np.argsort(np.array(data[key]['recons'][norm_key]))
        sort_param = p_indices[::-1][:end]
        print('\n p_indices: ', p_indices)
        species = tuple([set([int(j)+1 for j in data[key]['labels'][i]]) for i in sort_param])  # ("Adelie", "Chinstrap", "Gentoo")
        #print(data[key]['recons'][norm_key])
        print('\n species: ', species)
        values = data[key][name1][norm_key] + data[key]['recons'][norm_key]
        penguin_means = {
            "Anova-term": tuple([data[key]['recons'][norm_key][i] for i in sort_param]),#(38.79, 48.83, 47.50),
            name_partial: tuple([data[key][name1][norm_key][i] for i in sort_param]),  # (18.35, 18.43, 14.98),
            #'Groundtruth': (189.95, 195.82, 217.19),
        }
        print(species)
        print(penguin_means.items())
        x = np.arange(len(species))  # the label locations
        width = 0.2 #Grad=0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(layout='constrained')
        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            print(attribute)
            if attribute == 'Gradient' or attribute =='Hessian':
                if norm == "\infty":
                    label_ = r"$||\partial_{\mathbf{v}}f||_{\infty}$"
                else:
                    label_ = r"$||\partial_{\mathbf{v}}f||_1$"
                rects = ax.bar(x + offset, measurement, width, label=label_)
            else:
                if norm == "\infty":
                    label_ = r"$||f_{\mathbf{v}_{\max}, A}||_{\infty}$"
                else:
                    label_ = r"$||f_{\mathbf{v}_{\max}, A}||_1$"
                rects = ax.bar(x + offset, measurement, width, label=label_)
            #rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.legend(ncol=1)

            #print('rects: ', [list(it) for it in list(rects)])
            #ax.bar_label(rects, fmt='%.7f')
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        #ax.set_ylabel('$L_{}$-Norm'.format(norm))
        #ax.set_ylabel('$p = {}$'.format(norm))
        ax.axhline(y=clamp, color='r', linestyle='dashed')

        if cov is not None:
            ax.set_xlabel('Couplings $\mathbf{v}$')
        #ax.set_title('ANOVA terms vs partial derivatives')
        ax.set_xticks(x + width, species)
        ax.legend(loc='upper right', ncols=3)
        ax.set_yscale('log')
        ax.set_ylim(0, 3*max(values)/2)

        plt.show()

if __name__ == '__main__':

    output_folder = '/homes/numerik/fatimaba/store/Github/trafo_nova/Anova_AE/Output_files'
    cov = None
    run_man_opt = False#True
    tr_model = False
    evaluate = False
    all_upper_set =False
    #all_upper = '_all' if all_upper_set else ''
    #h_size = 1
    if tr_model:
        datas = set_gt([func2(return_f=False), func3(return_f=False)])
        train_model(datas, output_folder, h_size=1, N_epochs=int(5e4), run_man_opt=run_man_opt)
        train_model(datas, output_folder, N_epochs=int(5e4), run_man_opt=not run_man_opt)
        #datas = train_data(datas, output_folder, h_size=h_size, N_epochs=10000,
        #                                       cov=cov, run_man_opt=run_man_opt)
    else:
        import copy
        h_sizes = [1]#[1, 0.5, 0.25, 0.125]
        for j in range(0, 2):
            for h_size in h_sizes:
                print("\n .................................H_size: ", h_size)
                suff_ = '_cov_{}'.format(cov) if cov is not None else ''
                if not run_man_opt:
                    name = '{}/Test_anova{}_h_size_{}.json'.format(output_folder, suff_, h_size)
                else:
                    name ='{}/Test_anova{}_mo.json'.format(output_folder, suff_)
                with open(name) as convert_file:
                    datas = copy.deepcopy(json.load(convert_file))

                print(datas[str(j)]['groundtruth'])
                dim = datas[str(j)]['dim']
                #print(datas['0'])
                Rot_la, Rot_re = get_total_Rot(datas, str(0), man_opt=run_man_opt)
                hessian_supp = torch.as_tensor(datas['0']['svd_basis'])
                supp = hessian_supp.shape[1]
                N = hessian_supp.shape[0]
                #hessians = torch.zeros((N, dim, dim))
                #hessians[:, :supp, :supp] = hessian_supp
                x_test = torch.as_tensor(datas[str(j)]['x_test'])
                ground_truth = datas[str(j)]['groundtruth']
                ground_truth['v'] = torch.as_tensor(ground_truth['v'])
                U = torch.as_tensor(datas[str(j)]['grad_U'])
                #hessian = torch.as_tensor(datas[str(j)]['hessian'])
                #gradient = compute_gradient_autograd(torch.as_tensor(datas[str(j)]['x_test']), ground_truth)
                Rot = Rot_re.T
                #compare_anova_der(ground_truth, Rot.cpu().to(torch.float32))


                #get_vanishing_indices(ground_truth, gradient, hessian, Rot, U)
                if evaluate:
                    gradient = compute_gradient_autograd(x_test, ground_truth)
                    hessian = compute_hessian_auto(x_test, ground_truth)
                    results_ = get_vanishing_indices(ground_truth, gradient, hessian, Rot)
                    print(results_)
                    suff_ = '_cov_{}'.format(cov) if cov is not None else ''

                    if not run_man_opt:
                        with open('{}/Anova_deriv{}_h_size_{}_{}.json'.format(output_folder, suff_, h_size, j),
                                  'w') as convert_file:
                            json.dump(results_, convert_file, cls=NumpyEncoder)
                    else:
                        with open('{}/Anova_deriv{}_mo_{}.json'.format(output_folder, suff_, j), 'w') as convert_file:
                            json.dump(results_, convert_file, cls=NumpyEncoder)
                else:
                    suff_ = '_cov_{}'.format(cov) if cov is not None else ''
                    fname = None
                    if not run_man_opt:
                        fname = '{}/Anova_deriv{}_h_size_{}_{}.json'.format(output_folder, suff_, h_size, j)
                        with open(fname) as convert_file:
                            results_ = copy.deepcopy(json.load(convert_file))
                    else:
                        fname = '{}/Anova_deriv{}_mo_{}.json'.format(output_folder, suff_, j)
                        with open(fname) as convert_file:
                            results_ = copy.deepcopy(json.load(convert_file))
                    print('\n Filename: ', fname)
                    plot_histogram(results_, j=str(j), cov=cov)
