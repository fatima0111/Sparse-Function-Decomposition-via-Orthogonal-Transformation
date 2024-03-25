import copy
import itertools
import random
import geoopt
from geoopt.optim import RiemannianSGD
from landing import LandingSGD
import numpy as np
from torch.optim.lr_scheduler import MultiplicativeLR
import torch
import time
import math
from Libs.Rotations import compute_rot


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def stiefel_manifold_opt(hessian, init_rot=None, n_epochs_=int(2e4),
                         n_inits=5, n_init_epochs=int(5e3), print_mode=False,
                         opt_method='both', learning_rate=5e-4):
    '''
    Running RiemannianSGD or LandingSDG with init rotation matrix init_rot obtained from the Grid-search
    method 
    :param hessian: H of dimension Nxnxn where N is the number of sample points
    :param rand_rot: ground-truth orthogonal matrix A
    :param n_epochs_:
    :param device:s
    :return:
    '''
    hessian = hessian.to(device)
    lmbda = lambda epoch: .95
    D = hessian.shape[1]
    if opt_method == 'both':
        method_names = ["LandingSGD", "RiemannianSGD"]
        methods = [LandingSGD, RiemannianSGD]
        learning_rates = [learning_rate, learning_rate]
    else:
        method_names = [opt_method]
        methods = methods_dic[method]
        learning_rates = [learning_rate]
    methods_n_epochs = [n_epochs_, n_epochs_]
    sol = torch.randn(len(methods), D, D)
    index = 0

    param_mins = []
    best_losses = []
    losses_array = [[], []]
    times_ = []
    for method_name, method, learning_rate, n_epochs in zip(method_names, methods, learning_rates, methods_n_epochs):
        t1 = time.time()
        iterates = []
        if init_rot is not None:
            init_weights = init_rot.to(device)
            init_time = None
        else:
            init_weights, init_time = init_param(
            hessian, method=method, D=D, n_inits=n_inits, n_init_epochs=n_init_epochs, print_mode=print_mode)
        param = geoopt.ManifoldParameter(
            init_weights.clone(), manifold=geoopt.Stiefel(canonical=False)
        )
        param_min = param.clone().detach()
        test_min = 1e5
        optimizer = method((param,), lr=learning_rate)
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        epoch = 0
        with torch.no_grad():
            res = torch.matmul(torch.matmul(param, hessian), param.t())
            res2 = ((res.abs() ** 2).mean(dim=0) + 1e-8).sqrt()  # **.5
            loss = torch.norm(res2, p=1)
        xi_norm = 3
        while epoch <= n_epochs and xi_norm > 1e-17:
            loss1 = loss.clone().detach()
            optimizer.zero_grad()
            res = torch.matmul(torch.matmul(param, hessian), param.t())

            res2 =((res.abs()**2).mean(dim=0)+1e-8).sqrt()
            loss = torch.norm(res2, p=1)

            if loss.clone().item() < test_min:
                test_min = loss.clone().item()
                param_min = param.clone().detach()
            loss.backward()
            with torch.no_grad():
                xi = param.grad-.5*param@(param.T@param.grad + param.grad.T@param)
                xi_norm = torch.norm(xi, p='fro')
                if epoch % 500 == 0 and print_mode:
                    print("max(1, torch.norm(param.grad, p='fro')): ", torch.norm(xi, p='fro'))
            optimizer.step()
            if epoch % 700 == 0 and scheduler.get_last_lr()[0] > 1e-6:
                if print_mode:
                    print('loss1-loss: ', abs(loss1 - loss))
                    print(epoch, loss.item())
                scheduler.step()
                if print_mode:
                    print('******************* Learning_rate: {}'.format(scheduler.get_last_lr()[0]))
            if epoch <= int(1e3) and epoch % 50 == 0:
                losses_array[index].append(test_min)

            elif epoch > int(1e3) and epoch % 1000 == 0:
                losses_array[index].append(test_min)
                #iterates.append(param.data.clone())
            epoch += 1
        if init_time is None:
            times_.append(time.time() - t1)
        else:
            times_.append([init_time, time.time() - t1])

        #sol[index, :, :] = iterates[-1]
        param_mins.append(param_min)
        best_losses.append(test_min)
        if print_mode:
            print('')
            print('test_min: {}'.format(test_min))
            print('')
        index += 1

        distance_list = []
        for matrix in iterates:
            d = (
                torch.norm(matrix.matmul(matrix.transpose(-1, -2)) - torch.eye(D))
            )
            distance_list.append(d.item())

    sol[0, :, :] = param_mins[0]
    sol[1, :, :] = param_mins[1]
    return [param_mins, losses_array, times_]

def init_param(hessian, method=RiemannianSGD, D=5, n_inits=5, n_init_epochs=int(5e3),
               learning_rate=5e-4, print_mode=False):
    init_params = []
    losses = []
    losses2 = []
    lmbda = lambda epoch: .95
    hessian = hessian
    t1 = time.time()
    alphas = np.random.uniform(0, 2 * math.pi, size=(n_inits, int(D * (D - 1) / 2)))
    alphas = torch.as_tensor(alphas)
    init_weights = compute_rot(D, alphas)
    for n_init in range(n_inits):
        init_weight = init_weights[n_init, :, :]
        param = geoopt.ManifoldParameter(
            init_weight.clone(), manifold=geoopt.Stiefel(canonical=False)
        )
        param_min = param.clone().detach()
        test_min = 1e5
        test_min2 = 1e7
        optimizer = method((param,), lr=learning_rate)
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        epoch = 0
        xi_norm = 3
        while epoch <= n_init_epochs and xi_norm>1e-17:
            optimizer.zero_grad()
            res = torch.matmul(torch.matmul(param, hessian), param.t())
            res2 = ((res.abs() ** 2).mean(dim=0)+1e-8).sqrt()
            loss = res2.sum()
            if loss.item() < test_min2:
                test_min = loss.item()
                test_min2 = (((torch.matmul(torch.matmul(param, hessian), param.t()).abs()**2).mean(dim=0)**.25).sum()**2)
                param_min = param.clone().detach()
            loss.backward()
            with torch.no_grad():
                xi = param.grad-.5*param@(param.T@param.grad + param.grad.T@param)
                xi_norm = torch.norm(xi, p='fro')
            optimizer.step()
            if epoch % 500 == 0:
                if print_mode:
                    print(epoch, loss.item(), (((torch.matmul(torch.matmul(param, hessian), param.t()).abs()**2).mean(dim=0)**.25).sum()**2))
                    scheduler.step()
                    print('******************* Learning_rate: {}'.format(scheduler.get_last_lr()[0]))
            epoch += 1
        losses.append(test_min)
        losses2.append(test_min2)
        init_params.append(param_min)
    init_time = time.time() - t1
    if print_mode:
        print('\n losses: ', losses)
        print('\n losses: ', losses2)
        print('opt_loss: {}, ind_min_loss: {} opt_loss2: {}, ind_min_loss2: {}'.format(
            min(losses), losses.index(min(losses)), min(losses2), losses2.index(min(losses2))))
    return init_weights[losses2.index(min(losses2)), :, :], init_time

def sprod(sym, cop, ort, ort_shape):
    ort = np.reshape(ort, ort_shape)
    n_sym = sym.shape[0]
    n_cop = len(cop)
    print(ort.shape)
    n = max(len(cop)* sym.shape[0] + ort.shape[0]**2, ort.shape[0]*ort.shape[1])
    result = np.zeros(n)
    for ind in range(n_cop):
        i,j = cop[ind]
        result[ind*n_sym:(ind+1)*n_sym] = (ort[i].T @ sym @ ort[j]).flatten()
    ort_cond = ort @ ort.T - np.eye(ort.shape[0])
    result[len(cop)*n_sym:(len(cop)*n_sym)+ort.shape[0]**2] = ort_cond.flatten()
    return result


def generate_cop(dim=5, s=5):
    couplings = list(itertools.product(range(dim), range(dim)))
    sorted_couplings = [tuple(u) for u in list(map(sorted, couplings))]
    candidates = [list(u) for u in list(set(sorted_couplings))]
    U = random.sample(candidates, s)#[0]
    print('\n dim={} s {}'.format(dim, s), U)
    return U #U #[list(i) for i in set(cops)]
def generate_block_components(dim, K, max_subfunction_entries, max_components=4,
                              probs=None, add_diag=True, dense=None):

    blocks = []
    assert (K * max_components <= dim)
    print(probs, len(range(2, max_components+1)))
    assert (probs is None or len(probs) == len(range(2, max_components+1)))
    source = list(range(0, dim))
    U = []
    for k in range(K):# K= Number of subgraphs
        ds = np.random.choice(range(2, max_components+1), p=probs)
        block = np.random.choice(source, ds, replace=False)#number of node in the subgrad G_k
        source1 = [source[i] for i in range(len(source)) if source[i] not in block]
        source = source1
        couplings = list(itertools.product(block, block))
        sorted_couplings = [tuple(u) for u in list(map(sorted, couplings))]
        couplings = [list(u) for u in list(set(sorted_couplings))]
        nonhomogeneous_coupling = [u for u in couplings if u[0] != u[1]]
        homogeneous_couplings = [u for u in couplings if u[0] == u[1]]
        copy_cop = copy.deepcopy(nonhomogeneous_coupling)
        U_k=[]
        num_ = {}
        for j in block:
            num_[j] = 0
        N_block = 1 if ds == 2 else len(block)-1
        while len(U_k) < N_block:
            ind_ = np.random.choice(list(range(len(
                copy_cop))), 1)[0]
            inner_u = copy_cop[ind_]
            if num_[inner_u[0]]<2 and num_[inner_u[1]]<2:
                U_k.append(inner_u)
                num_[inner_u[0]] += 1
                num_[inner_u[1]] += 1
            copy_cop.remove(inner_u)
        dense_ = bool(np.random.randint(1)) if dense is None else dense
        if dense_:
            rest = [u for u in nonhomogeneous_coupling if u not in U_k]
            ds_inner = np.random.choice(1, len(rest)//2)
            indices = np.random.choice(list(range(len(rest))), ds_inner)
            if ds_inner != 0:
                U_k += [rest[i] for i in indices]
        block = list(set().union(*U_k))
        if add_diag:
            n_block = np.random.randint(len(set([l for v in U_k for l in v])))
            n_diag = np.random.choice(list(set().union(*homogeneous_couplings)), size=n_block, replace=False)
            for n in range(n_block):
                U_k.append([n_diag[n], n_diag[n]])
        blocks.append(block)
        U += U_k
    print('blocks: ', blocks, "U: ", U)
    return U, blocks


def random_w_pattern(cop, dim=5):
    '''

    :param cop:
    :param dim:
    :return:
    '''
    a = 2*torch.rand(size=(dim, dim))-1
    a = torch.mm(a, a.t())
    for i in range(dim):
        for j in range(i, dim):
            if [i, j] not in cop:
                a[i, j] = 0
                a[j, i] = 0

    return a.cpu().numpy()


def ran_p(cop, dim=5, n=None, v=None):
    '''

    :param cop:
    :param dim:
    :param n:
    :param epsilon:
    :param v:
    :return:
    '''
    if v is None:
        v = np.linalg.svd(np.random.rand(dim, dim))[0]
    else:
        v = v.cpu().numpy()
    if n is not None:
        a = 2 * torch.rand(size=(n, dim, dim)) - 1
        a = a + torch.permute(a, (0, 2,1))
        for i in range(dim):
            for j in range(i, dim):
                if [i, j] not in cop and [j, i] not in cop:
                    a[:, i, j] = 0
                    a[:, j, i] = 0
        return np.float32(v.T @ a.cpu().numpy() @ v), np.float64(v)
    else:
        return np.float64(v)

def get_rank_svd(diag, dim, eps=3, is_hessian=True):
    """
    :param diag : d-dimensional vector
    :param eps:
    :return:
    """
    diag1 = np.floor(np.log10(np.abs(diag[:-1].cpu().detach().numpy()))).astype(int)
    diag2 = np.floor(np.log10(np.abs(diag[1:].cpu().detach().numpy()))).astype(int)
    diff = np.where((diag1-diag2) >= eps)[-1]
    #print("\n diff: ", diag1-diag2, diff, dim)
    if len(diff) >=1:
        diff = int(diff[0])+1
    else:
        if is_hessian and dim == math.isqrt(dim) ** 2:
            dim_ = dim ** .5
            diff = int(dim_ * (dim_ + 1) / 2)
        else:
            diff =dim
    return int(diff)

