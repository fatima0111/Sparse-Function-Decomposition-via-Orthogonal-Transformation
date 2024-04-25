import geoopt
from geoopt.optim import RiemannianSGD
from landing import LandingSGD
import numpy as np
from torch.optim.lr_scheduler import MultiplicativeLR
import torch
import time
import math
from Libs.Rotations import compute_rotation_U
from Utils.Evaluation_utils import loss_function12, loss_function_sparse

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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
        methods = [LandingSGD if opt_method == LandingSGD else RiemannianSGD]
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
            init_weights, init_time = random_init(
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
            #res = torch.matmul(torch.matmul(param, hessian), param.t())
            #res2 = ((res.abs() ** 2).mean(dim=0) + 1e-8).sqrt()  # **.5
            loss = loss_function12(param, hessian, eps=1e-8)#torch.norm(res2, p=1)
        xi_norm = 3
        while epoch <= n_epochs and xi_norm > 1e-17:
            loss1 = loss.clone().detach()
            optimizer.zero_grad()
            #res = torch.matmul(torch.matmul(param, hessian), param.t())

            #res2 =((res.abs()**2).mean(dim=0)+1e-8).sqrt()
            loss = loss_function12(param, hessian, eps=1e-8) #torch.norm(res2, p=1)

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
            epoch += 1
        if init_time is None:
            times_.append(time.time() - t1)
        else:
            times_.append([init_time, time.time() - t1])
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

def random_init(hessian, method=RiemannianSGD, D=5, n_inits=5, n_init_epochs=int(5e3),
                learning_rate=5e-4, print_mode=False):
    init_params = []
    losses = []
    losses2 = []
    lmbda = lambda epoch: .95
    hessian = hessian
    t1 = time.time()
    alphas = np.random.uniform(0, 2 * math.pi, size=(n_inits, int(D * (D - 1) / 2)))
    alphas = torch.as_tensor(alphas)
    init_weights = compute_rotation_U(D, alphas)
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
        while epoch <= n_init_epochs and xi_norm > 1e-17:
            optimizer.zero_grad()
            #res = torch.matmul(torch.matmul(param, hessian), param.t())
            #res2 = ((res.abs() ** 2).mean(dim=0)+1e-8).sqrt()
            loss = loss_function12(param, hessian, eps=1e-8)#res2.sum()
            if loss.item() < test_min2:
                test_min = loss.item()
                test_min2 = loss_function_sparse(param, hessian)
                #(((torch.matmul(torch.matmul(param, hessian), param.t()).abs()**2).mean(dim=0)**.25).sum()**2)
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



