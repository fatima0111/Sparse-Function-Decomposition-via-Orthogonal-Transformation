import json
from os.path import dirname, abspath
import numpy
import torch
import numpy as np

from matplotlib import pyplot as plt
import math
import matplotlib
from Utils.Function_utils import compute_gradient_autograd

############################################
def plot_f1(a, k=np.sqrt(2)/2):
    '''

    :param a:
    :param k:
    :return:
    '''
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    U = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    st = 2 * math.pi + math.pi
    x, z = np.meshgrid(np.arange(-st, st, math.pi / 32),
                       np.arange(-st / 4, st / 4, math.pi / 128))
    X = x.T
    Z = z.T
    Y = 0 * np.ones((X.shape[0], X.shape[0])) + math.pi / 8
    colourmap_plane = np.full(X.shape + (4,), [0, 0.2, 0.2, .1])
    r = np.arange(-st, st, math.pi / 32)
    x1, x2 = np.meshgrid(r, r)
    xy = np.array([x1.flatten(), x2.flatten()]).T
    X_bar = np.vstack([X[-1, :], x1[0, :]])
    Y_bar = np.vstack([Y[-1, :], x2[0, :]])

    X_surf = np.vstack([X, X_bar, x1])
    Y_surf = np.vstack([Y, Y_bar, x2])
    def f_(U_, rot_x=False):
        def f_g(val):
            val = val.T
            if not rot_x:
                return 0*np.sin(k * U_[:, 0] @ val) + np.sin(k * U_[:, 1] @ val)
            else:
                return 0*np.sin(k * U_[:, 0] @ U_ @ val) + np.sin(k * (U_[:, 1] @ U_ @ val))
        return f_g
    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={'projection': '3d'})
    f = f_(U)
    func = f(xy).reshape(x1.shape)
    colormap_func = plt.get_cmap('viridis')((func - func.min()) / func.ptp())
    Z_bar = np.vstack([Z[-1, :], func[0, :]])
    colormap_bar = np.full(Z_bar.shape + (4,), [1, 1, 1, 0])
    Z_surf = np.vstack([Z, Z_bar, func])
    C_surf = np.vstack([colourmap_plane, colormap_bar, colormap_func])
    ax.plot_surface(X_surf, Y_surf, Z_surf, facecolors=C_surf, rstride=1, cstride=1)
    print(X_surf.shape, Y_surf.shape, Z_surf.shape, C_surf.shape)
    #ax.set_xlabel(r"$x_{1}$", font="CMU Serif", fontsize=20)
    #ax.set_ylabel(r"$x_{2}$", font="CMU Serif", fontsize=20)
    ax.set_xticks(np.arange(-10, 10 + 1, 5.0))
    ax.set_yticks(np.arange(-10, 10 + 1, 5.0))
    ax.set_zlabel(r'$f^1(x_{1},x_{2})$', font="CMU Serif", fontsize=30)
    plt.tight_layout()
    plt.savefig(root + '/f1.png')
    ##################################### Rotate input space #############################
    fig, ax = plt.subplots(figsize=(11, 11),
                           subplot_kw={'projection': '3d'})
    f = f_(U, rot_x=True)
    func = f(xy).reshape(x1.shape)
    colormap_func = plt.get_cmap('viridis')((func - func.min()) / func.ptp())
    Z_bar = np.vstack([Z[-1, :], func[0, :]])
    colormap_bar = np.full(Z_bar.shape + (4,), [1, 1, 1, 0])
    Z_surf = np.vstack([Z, Z_bar, func])
    C_surf = np.vstack([colourmap_plane, colormap_bar, colormap_func])
    ax.plot_surface(X_surf, Y_surf, Z_surf, facecolors=C_surf, rstride=1, cstride=1)
    print(X_surf.shape, Y_surf.shape, Z_surf.shape, C_surf.shape)
    #ax.set_xlabel(r"$x_{1}$", font="CMU Serif", fontsize=20)
    #ax.set_ylabel(r"$x_{2}$", font="CMU Serif", fontsize=20)
    ax.set_xticks(np.arange(-10, 10 + 1, 5.0))
    ax.set_yticks(np.arange(-10, 10 + 1, 5.0))
    ax.set_zlabel(r'$f_U^1(x_{1},x_{2})$', font="CMU Serif", fontsize=30)
    plt.tight_layout()
    plt.savefig(root + '/f1_U.png')

def plot_f2(a, k=6):
    U = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    X, Y = np.meshgrid(x, y)
    xy = np.array([X.flatten(), Y.flatten()]).T
    x_gr = np.linspace(-1, 1, 50)
    y_gr = np.linspace(-1, 1, 50)
    X_gr, Y_gr = np.meshgrid(x_gr, y_gr)
    xy_gr = np.array([X_gr.flatten(), Y_gr.flatten()]).T
    def func(U_, rot_x=False):
        if type(U_) == numpy.ndarray:
            U_ = torch.as_tensor(U_)
        def f_g(val):
            val = val.T
            if type(val) == numpy.ndarray:
                val = torch.as_tensor(val)
            if not rot_x:
                return torch.sin(k * U_[:, 0] @ val) + torch.sin(k * U_[:, 1] @ val)
            else:
                return torch.sin(k * U_[:, 0] @ U_ @ val) + torch.sin(k * (U_[:, 1] @ U_ @ val))
        return f_g

    Z_f = func(U)
    Z = Z_f(xy).cpu().detach().numpy()
    print("Z.shape: ", Z.shape)
    Z = Z.reshape(X.shape)
    grad = compute_gradient_autograd(xy_gr, func=func(U)).cpu().detach().numpy()
    grad = grad[0, :].reshape(X_gr.shape)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
    CJ = plt.get_cmap('viridis')((Z - Z.min()) / Z.ptp())
    ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, facecolors=CJ)
    ax.set_zlim3d(-1.5, 1.5)
    #ax.set_xlabel(r"$x_{1}$", fontsize=16)
    #ax.set_ylabel(r"$x_{2}$", fontsize=16)
    ax.set_zlabel(r'$f^2(x_1, x_2)$', font="CMU Serif", fontsize=25)
    ax.set_xticks(np.arange(-1, 1 + 1, 0.5))
    ax.set_yticks(np.arange(-1, 1 + 1, 0.5))
    plt.savefig(root + '/f2.png')
    ###### Gradient
    fig = plt.figure(figsize=(6, 6))
    cs = plt.contourf(X_gr, Y_gr, grad, 20, cmap='RdGy')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r"$\partial_{x_1}f^2(x_1, x_2)$", font="CMU Serif", fontsize=30)
    plt.xticks(np.arange(-1, 1.02, 0.5))
    plt.yticks(np.arange(-1, 1.02, 0.5))
    #plt.xlabel(r"$x_{1}$", fontsize=16)
    #plt.ylabel(r"$x_{2}$", fontsize=16)
    plt.savefig(root + '/f2_grad_x1.png')

    Z_fU = func(U, rot_x=True)
    Z_U = Z_fU(xy).detach().cpu().detach().numpy()
    Z_U = Z_U.reshape(X.shape)
    grad_U = compute_gradient_autograd(xy_gr, func=func(U, rot_x=True)).cpu().detach().numpy()
    grad_U = grad_U[0, :].reshape(X_gr.shape)

    fig, ax = plt.subplots(figsize=(6, 6),  subplot_kw={'projection': '3d'})
    CJ = plt.get_cmap('viridis')((Z_U - Z_U.min()) / Z_U.ptp())
    ax.plot_surface(X, Y, Z_U, cmap='viridis', rstride=1, cstride=1, facecolors=CJ)
    ax.set_zlim3d(-1.5, 1.5)
    #ax.set_xlabel(r"$x_{1}$", fontsize=16)
    #ax.set_ylabel(r"$x_{2}$", fontsize=16)
    ax.set_zlabel(r'$f_U^2(x_1, x_2)$', font="CMU Serif", fontsize=25)
    ax.set_xticks(np.arange(-1, 1 + 1, 0.5))
    ax.set_yticks(np.arange(-1, 1 + 1, 0.5))
    plt.savefig(root + '/f2_U.png')

    fig = plt.figure(figsize=(6, 6))
    cs = plt.contourf(X_gr, Y_gr, grad_U, 20, cmap='RdGy')
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r"$\partial_{x_1}f_U^2(x_1, x_2)$", font="CMU Serif", fontsize=30)
    plt.xticks(np.arange(-1, 1.02, 0.5))
    plt.yticks(np.arange(-1, 1.02, 0.5))
    #plt.xlabel(r"$x_{2}$", fontsize=16)
    #plt.ylabel(r"$x_{2}$", fontsize=16)
    plt.savefig(root + '/f2_U_grad_x1.png')

if __name__ == '__main__':
    a = math.pi/4
    root = dirname(dirname(abspath(__file__))) + '/Plots/Bivariate_functions'
    print(root)
    plot_f1(a)
    plot_f2(a)
    plt.show()