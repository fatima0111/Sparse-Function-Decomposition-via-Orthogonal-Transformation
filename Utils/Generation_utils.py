import numpy as np
import torch
import copy
import itertools
import random
import math

batches_random_matrices = {
    2: {1.0: math.pi,
        1 / 2: math.pi,
        1 / 4: math.pi,
        1 / 8: math.pi,
        1/10: math.pi},
    3: {1.0: math.pi / 2,
        1 / 2: math.pi / 2,
        1 / 4: math.pi / 2,
        1 / 8: math.pi / 2,
        1/10: math.pi/2},
    4: {1.0: math.pi / 2,
        1 / 2: math.pi / 2,
        1 / 4: math.pi / 2,
        1 / 8: math.pi/2,
        1/10: 1.0},
    5: {1: 2.36}
}

batches_random_functions = {
        2: {3: math.pi,
            1: math.pi,
            1/2: math.pi,
            1/4: math.pi,
            1/8: math.pi},
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
def generate_cop(d=5, s=5):
    '''
    Generates a set of jointly non-sparse entries J
    :param d: dimension
    :param s: number of sparsity patterns
    :return: J
    '''
    couplings = list(itertools.product(range(d), range(d)))
    sorted_couplings = [tuple(u) for u in list(map(sorted, couplings))]
    candidates = [list(u) for u in list(set(sorted_couplings))]
    J = random.sample(candidates, s)#[0]
    print('\n d={} s {}'.format(d, s), J)
    return J
def generate_bloc_components(d, K, max_bloc_size=4,
                              probs=None, add_diag=True, dense=None):
    '''
    Generates randomly a set of jointly non-sparse entries having a bloc form according to
    :param d: dimension
    :param K: number of blocs
    :param max_bloc_size: maximal size of the blocs
    :param probs:
    :param add_diag:
    :param dense:
    :return: J
    '''
    blocs = []
    assert (K * max_bloc_size <= d)
    print(probs, len(range(2, max_bloc_size+1)))
    assert (probs is None or len(probs) == len(range(2, max_bloc_size+1)))
    source = list(range(0, d))
    J = []
    for k in range(K):# K= Number of subgraphs
        ds = np.random.choice(range(2, max_bloc_size+1), p=probs)
        bloc = np.random.choice(source, ds, replace=False)
        source1 = [source[i] for i in range(len(source)) if source[i] not in bloc]
        source = source1
        couplings = list(itertools.product(bloc, bloc))
        sorted_couplings = [tuple(u) for u in list(map(sorted, couplings))]
        couplings = [list(u) for u in list(set(sorted_couplings))]
        nonhomogeneous_coupling = [u for u in couplings if u[0] != u[1]]
        homogeneous_couplings = [u for u in couplings if u[0] == u[1]]
        copy_cop = copy.deepcopy(nonhomogeneous_coupling)
        J_k=[]
        num_ = {}
        for j in bloc:
            num_[j] = 0
        N_bloc = 1 if ds == 2 else len(bloc)-1
        while len(J_k) < N_bloc:
            ind_ = np.random.choice(list(range(len(copy_cop))), 1)[0]
            inner_u = copy_cop[ind_]
            if num_[inner_u[0]] < 2 and num_[inner_u[1]] < 2:
                J_k.append(inner_u)
                num_[inner_u[0]] += 1
                num_[inner_u[1]] += 1
            copy_cop.remove(inner_u)
        dense_ = bool(np.random.randint(1)) if dense is None else dense
        if dense_:
            rest = [u for u in nonhomogeneous_coupling if u not in J_k]
            ds_inner = np.random.choice(1, len(rest)//2)
            indices = np.random.choice(list(range(len(rest))), ds_inner)
            if ds_inner != 0:
                J_k += [rest[i] for i in indices]
        bloc = list(set().union(*J_k))
        if add_diag:
            n_bloc = np.random.randint(len(set([l for v in J_k for l in v])))
            n_diag = np.random.choice(list(set().union(*homogeneous_couplings)), size=n_bloc, replace=False)
            for n in range(n_bloc):
                J_k.append([n_diag[n], n_diag[n]])
        blocs.append(bloc)
        J += J_k
    print('blocs: ', blocs, "J: ", )
    return J, blocs


def random_w_pattern(J, d=5):
    '''
    Generates symmetric matrix $H$ with uniform distributed entries such that for all i,j=1,..., d
            H_ij ~ Unif([-1, 1]) if [i,j] or [j, i] in J
            H_ij = 0, otherwise
    :param J: set of jointly non-sparse entries
    :param d: dimension
    :return: return symmetric matrix H having z
    '''
    H = 2*torch.rand(size=(d, d))-1
    H = torch.mm(H, H.t())
    for i in range(d):
        for j in range(i, d):
            if [i, j] not in J:
                H[i, j] = 0
                H[j, i] = 0
    return H.cpu().numpy()


def ran_p(J, d=5, N=None, R=None):
    '''
    Generates symmetric matrix $H$ with uniform distributed entries such that for all i,j=1,..., d
            H_ij ~ Unif([-1, 1]) if [i,j] or [j, i] in J
            H_ij = 0, otherwise
    :param J: set of jointly non-sparse entries
    :param d: dimension of the symmetric matrices H
    :param N: number of symmetric matrices H
    :param R: Orthogonal matrix of dimension dxd
    :return: Set of non-sparse \mathcal H_R(J) = R.T\mathcal H(J) R = {R.T H^n R: n=1, ..., N} if R is not None else R
    '''
    if R is None:
        R = np.linalg.svd(np.random.rand(d, d))[0]
    else:
        R = R.cpu().numpy()
    if N is not None:
        H = 2 * torch.rand(size=(N, d, d)) - 1
        H = H + torch.permute(H, (0, 2, 1))
        for i in range(d):
            for j in range(i, d):
                if [i, j] not in J and [j, i] not in J:
                    H[:, i, j] = 0
                    H[:, j, i] = 0
        return np.float64(R.T @ H.cpu().numpy() @ R), np.float64(R)
    else:
        return np.float64(R)

def get_rank_svd(diag, d, eps=3, is_hessian=True):
    """
    Determines the gap
    :param diag : d-dimensional vector
    :param eps:
    :return:
    """
    diag1 = np.floor(np.log10(np.abs(diag[:-1].cpu().detach().numpy()))).astype(int)
    diag2 = np.floor(np.log10(np.abs(diag[1:].cpu().detach().numpy()))).astype(int)
    diff = np.where((diag1-diag2) >= eps)[-1]
    #print("\n diff: ", diag1-diag2, diff, dim)
    if len(diff) >= 1:
        diff = int(diff[0])+1
    else:
        if is_hessian and d == math.isqrt(d) ** 2:
            d_ = d ** .5
            diff = int(d_ * (d_ + 1) / 2)
        else:
            diff = d
    return int(diff)