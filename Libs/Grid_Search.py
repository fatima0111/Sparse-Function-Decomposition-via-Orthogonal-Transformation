import sys
sys.path.append('/homes/math/ba/trafo_nova/')
sys.path.append('//')


import math
from Anova_AE.Libs.roots import stiefel_manifold_opt
from Anova_AE.Libs.Rotations import compute_rot, generate_angles_interval
import torch
#from pykeops.torch import LazyTensor
import numpy as np
import json
import itertools
import time
from enum import Enum


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Method(Enum):
    Manifold_Opt = 1
    Grid_Search = 2
    Manifold_Opt_GS = 3


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return json.JSONEncoder.default(self, obj)

class Node:
    def __init__(self, key=None, start_v=None, end_v=None):
        '''

        :param key:
        :param start_v:
        :param end_v:
        '''
        self.key = key
        self.start_v = start_v
        self.end_v = end_v


class Graph:
    def __init__(self, d, batch_h=math.pi/2):
        '''

        :param p:
        :param batch_h:
        '''
        self.nodes = {}
        self.batch_h = batch_h
        self.bild_graph(d)
        #print('\n Bild graph: ', time.time()-t1)
        #print('self.nodes.keys(): ', len(self.nodes.keys()), self.nodes.keys())
        self.path = []

    def bild_graph(self, d, start=0):
        '''

        :param d: dimension of the space
        :param start:
        :return:
        '''
        p=int(d*(d-1)/2)
        angles = generate_angles_interval(d)
        len_batch_old = 0
        for i in range(p):
            end_i = angles[i]
            batch_points = np.arange(start, end_i + self.batch_h, self.batch_h).tolist()
            #print('batch_points: ', len(batch_points), batch_points, p, d, self.batch_h)
            #print('')
            self.nodes[i] = []
            for ind_j, j in enumerate(batch_points[:-1]):
                c_j = ind_j + len_batch_old #i*len(batch_points[:-1])+ind_j
                #print(c_j)
                end_value = batch_points[ind_j + 1] if batch_points[ind_j + 1] <= end_i else end_i
                c_node = Node(key=c_j, start_v=j, end_v=end_value)
                #self.nodes[c_j] = c_node
                self.nodes[i].append(c_node)
            len_batch_old += len(batch_points[:-1])

class Grid_Search:
    def __init__(self, datas=None, hessians=None, h=.5, batch_h=math.pi/2, block_form=False):
        '''

        :param hessian:
        :param v:
        :param h:
        :param batch_h:
        '''
        print('\n Start Grid: ')
        self.block_form = block_form
        self.h = h
        self.hessians = hessians
        if not self.block_form:
            assert (datas is not None)
            self.d = datas[0]['dim']
            self.p = int((self.d * (self.d - 1)) / 2)
            self.t_rot = 0
            self.t_losses = torch.zeros(len(datas))
            self.t_noisy_losses = torch.zeros(len(datas))
            self.min_losses = torch.ones(len(datas)) * 1e7
            self.min_losses_noisy = torch.ones(len(datas)) * 1e7
            self.min_rotations = [torch.eye(self.d) for j in range(len(datas))]
            self.min_rotations_noisy = [torch.eye(self.d) for j in range(len(datas))]
        else:
            assert(hessians is not None)
            self.hessians = hessians
            self.d = hessians[0].shape[1]
            self.p = int((self.d * (self.d - 1)) / 2)
            self.t_rot = 0
            self.t_losses = torch.zeros(len(hessians))
            self.t_noisy_losses = torch.zeros(len(hessians))
            self.min_losses = torch.ones(len(hessians)) * 1e7
            self.min_losses_noisy = torch.ones(len(hessians)) * 1e7
            self.min_rotations = [torch.eye(self.d) for j in range(len(hessians))]
            self.min_rotations_noisy = [torch.eye(self.d) for j in range(len(hessians))]

        self.graph = Graph(self.d, batch_h)
        nest_list = []
        for k in range(self.p):
            nest_list.append(self.graph.nodes[k])
        self.elem = itertools.product(*nest_list)
        self.n_paths = math.prod([len(nest_list[k]) for k in range(len(nest_list))])#self.n_c**self.p
        if not self.block_form:
            self.cover_grid(datas=datas)
        else:
            self.cover_grid(hessians=hessians)
    def loss(self, R, datas=None, hessians=None):
        '''
        :param R:
        :param hessian:
        :return:
        '''
        times = []
        times_noisy = []
        min_rotations = []
        min_rotations_noisy = []
        min_losses = []
        min_losses_noisy = []
        if self.block_form:
            assert(hessians is not None)
            for hessian in hessians:
                out1, _ = self.closs(R, hessian)
                times.append(out1[0])
                min_rotations.append(out1[1])
                min_losses.append(out1[2])
        else:
            assert(datas is not None)
            for data in datas:
                hessian = torch.as_tensor(data['svd_basis'], dtype=torch.float64)
                hessian_noisy = torch.as_tensor(data['svd_basis_noisis'], dtype=torch.float64)
                out1, out2 = self.closs(R, hessian, hessian_noisy)
                times.append(out1[0])
                min_rotations.append(out1[1])
                min_losses.append(out1[2])
                times_noisy.append(out2[0])
                min_rotations_noisy.append(out2[1])
                min_losses_noisy.append(out2[2])
        return [min_losses, min_rotations, times], [min_losses_noisy, min_rotations_noisy, times_noisy]
    def closs(self, R, hessian, hessian_noisy=None):
        t1 = time.time()
        mult1 = torch.matmul(
            R.reshape(R.shape[0], 1, R.shape[1], R.shape[1]),
            hessian.reshape(1, hessian.shape[0], hessian.shape[1], hessian.shape[1]))
        losses = ((torch.matmul(mult1,
                                R.reshape(R.shape[0], 1, R.shape[1], R.shape[1]).permute([0, 1, 3, 2])
                                ).abs() ** 2).mean(dim=1) ** .5).sum(dim=[1, 2])
        time1 = time.time() - t1
        ind = torch.argmin(losses)
        min_rot = R[ind, :, :].squeeze().clone()
        t2 = time.time()
        if hessian_noisy is not None:
            mult1_noisy = torch.matmul(
                R.reshape(R.shape[0], 1, R.shape[1], R.shape[1]),
                hessian_noisy.reshape(1, hessian_noisy.shape[0], hessian_noisy.shape[1], hessian_noisy.shape[1]))
            losses_noisy = ((torch.matmul(mult1_noisy,
                                          R.reshape(R.shape[0], 1, R.shape[1], R.shape[1]).permute([0, 1, 3, 2])
                                          ).abs() ** 2).mean(dim=1) ** .5).sum(dim=[1, 2])
            time_n = time.time() - t2
            ind_noisy = torch.argmin(losses_noisy)
            min_rot_noisy = R[ind_noisy, :, :].squeeze().clone()
            return [time1, min_rot, losses[ind]], [time_n, min_rot_noisy, losses_noisy[ind_noisy]]
        else:
            return [time1, min_rot, losses[ind]], []
    def parametrize_SOn(self, d):
        '''

        :param d:
        :return:
        '''
        if d <= 1 or d > 5:
            exit(1)
        parameters = [[0, 2 * math.pi]] * self.p
        return parameters

    def discretize_grid(self):
        '''
        :return:
        '''
        if type(self.h) != list:
            self.h = [self.h] * self.p
        else:
            if len(self.h) != self.p:
                exit(1)
        if self.h[0] > self.graph.batch_h:
            exit(1)
        sets = []
        for ind_i, i in enumerate(self.h):
            set_in = torch.arange(self.graph.path[ind_i].start_v, self.graph.path[ind_i].end_v+i,
                                   i) #.cpu()#dtype=torch.float
            set_in[-1] = set_in[-1] if set_in[-1] <= self.graph.path[ind_i].end_v else self.graph.path[ind_i].end_v
            sets.append(set_in)
        return torch.cartesian_prod(*sets)

    def update_rot(self, datas=None, hessians=None):
        '''
        :return:
        '''
        times_noisy_losses = None
        losses_noisy = None
        rotations_noisy = None
        N_datas = len(datas) if not self.block_form else len(hessians)
        t1 = time.time()
        points = self.discretize_grid()
        if points.ndim == 1:
            points = points.unsqueeze(dim=1)
        rotations = compute_rot(self.d, points)
        time_rot = time.time()-t1
        output, output_noisy = self.loss(rotations, datas=datas, hessians=hessians)
        losses, rotations, times_losses = output
        if not self.block_form:
            losses_noisy, rotations_noisy, times_noisy_losses = output_noisy
        for j in range(N_datas):
            if losses[j] < self.min_losses[j]:
                self.min_losses[j] = losses[j].clone()
                self.min_rotations[j] = rotations[j].squeeze().clone()
            if not self.block_form:
                if losses_noisy[j] < self.min_losses_noisy[j]:
                    self.min_losses_noisy[j] = losses_noisy[j].clone()
                    self.min_rotations_noisy[j] = rotations_noisy[j].squeeze().clone()
        if not self.block_form:
            return time_rot, times_losses, times_noisy_losses
        else:
            return time_rot, times_losses, []
    def check_condition(self):
        return False

    def cover_grid(self, datas=None, hessians=None):
        '''
        :param ManOpt:
        :param check_condition:
        :return:
        '''
        outer_batch = list(self.elem)
        for j in outer_batch:
            self.graph.path = list(j)
            time_rot, times_losses, times_noisy_losses = self.update_rot(datas=datas, hessians=hessians)
            self.t_rot += time_rot
            self.t_losses += torch.as_tensor(times_losses)
            if not self.block_form:
                self.t_noisy_losses += torch.as_tensor(times_noisy_losses)


def run_grid_search(datas, h=1/2, batch_h=math.pi / 2, convert_to_list=True):
    '''

    :param datas:
    :param h:
    :param batch_h:
    :param convert_to_list:
    :return:
    '''
    t1 = time.time()
    if convert_to_list:
        datas_list = [datas[j] for j in datas.keys()]
    else:
        datas_list = datas
    grid = Grid_Search(datas=datas_list, h=h, batch_h=batch_h, block_form=False)
    t2 = time.time()
    print('\n time: Grid Search : ', t2 - t1)
    for j in range(len(datas_list)):
        data = datas_list[j]
        j = int(j)
        result_grid_search = [grid.min_rotations[j], grid.min_losses[j]]
        result_grid_search_noise = [grid.min_rotations_noisy[j], grid.min_losses_noisy[j]]
        hessian = torch.as_tensor(data['hessian'], dtype=torch.float64)
        hessian_noise = torch.as_tensor(data['hessian_noise'], dtype=torch.float64)
        data['M']['gt']['Grid_search'] = (result_grid_search[0] @ hessian @ result_grid_search[0].T).abs().mean(
            dim=0)
        data['M']['noise']['Grid_search'] = (
                result_grid_search_noise[0] @ hessian_noise @ result_grid_search_noise[0].T).abs().mean(dim=0)
        data['R']['gt']['Grid_search'] = result_grid_search[0]
        data['R']['noise']['Grid_search'] = result_grid_search_noise[0]
        data['loss']['gt']['Grid_search'] = result_grid_search[1]
        data['loss']['noise']['Grid_search'] = result_grid_search_noise[1]
        data['time']['gt']['Grid_search'] = [grid.t_rot, grid.t_losses[j]]
        data['time']['noise']['Grid_search'] = [grid.t_rot, grid.t_noisy_losses[j]]
        data['h_size'] = h
        data['batch_h'] = batch_h
    return datas
def run_Man_Opt(data, v, optimizer_method=Method.Grid_Search,h=1/2, batch_h=math.pi / 2,
                N_epochs=int(1e4), print_mode=False, noisy_data=False):
    '''
    :param hessian:
    :param v:
    :param optimizer_method:
    :param h:
    :param batch_h:
    :param N_epochs:
    :return:
    '''
    hessian = torch.as_tensor(data['svd_basis'],
                              dtype=torch.float64
                              ) if not noisy_data else torch.as_tensor(
        data['svd_basis_noisis'], dtype=torch.float64)
    if optimizer_method == Method.Manifold_Opt:
        out = stiefel_manifold_opt(hessian, v, n_epochs_=N_epochs, print_mode=print_mode)
        return out
    else:
        if data['R']['gt']['Grid_search'] == {}:
            t1 = time.time()
            run_grid_search([data], h, batch_h, convert_to_list=False)
            t2 = time.time()
            print('\n time: Grid Search : ', t2 - t1)
        min_rot = torch.as_tensor(data['R']['gt']['Grid_search'],
                                  dtype=torch.float64
                                  ) if not noisy_data else torch.as_tensor(
                                data['R']['noise']['Grid_search'],
                                  dtype=torch.float64)
        if optimizer_method == Method.Manifold_Opt_GS:
            out = stiefel_manifold_opt(hessian, v, init_rot=min_rot, n_epochs_=N_epochs, print_mode=print_mode)
            Bs, losses, _ = out
            print('')
            print("\ndiff la: ", torch.norm(Bs[0] - min_rot), torch.norm(min_rot, p='fro'), torch.norm(Bs[0], p='fro'))
            print("\ndiff re: ", torch.norm(Bs[1] - min_rot), torch.norm(min_rot, p='fro'), torch.norm(Bs[1], p='fro'))
            print(losses[0][-1], losses[1][-1])
            var2 = out
            return var2, data
        elif optimizer_method == Method.Grid_Search:
            return data

def run_MO_Block(hessian, v, optimizer_method=Method.Grid_Search,h=1/2, batch_h=math.pi / 2,
                N_epochs=int(1e4), print_mode=False):
    '''
    :param hessian:
    :param v:
    :param optimizer_method:
    :param h:
    :param batch_h:
    :param N_epochs:
    :return:
    '''
    if optimizer_method == Method.Manifold_Opt:
        out = stiefel_manifold_opt(hessian, v, n_epochs_=N_epochs, print_mode=print_mode)
        return out
    else:
        grid = Grid_Search(hessians=[hessian], h=h, batch_h=batch_h, block_form=True)
        result_grid_search = [grid.min_rotations[0], grid.min_losses[0]]
        time_grid_search = [[grid.t_rot, grid.t_losses[0]]]
        min_rot = result_grid_search[0]
        if optimizer_method == Method.Manifold_Opt_GS:
            out = stiefel_manifold_opt(hessian, v, init_rot=min_rot, n_epochs_=N_epochs, print_mode=print_mode)
            Bs, losses, _ = out
            print('')
            print("\ndiff la: ", torch.norm(Bs[0] - min_rot), torch.norm(min_rot, p='fro'), torch.norm(Bs[0], p='fro'))
            print("\ndiff re: ", torch.norm(Bs[1] - min_rot), torch.norm(min_rot, p='fro'), torch.norm(Bs[1], p='fro'))
            print(losses[0][-1], losses[1][-1])
            var2 = out
            return var2, result_grid_search, time_grid_search
        elif optimizer_method == Method.Grid_Search:
            return result_grid_search, time_grid_search