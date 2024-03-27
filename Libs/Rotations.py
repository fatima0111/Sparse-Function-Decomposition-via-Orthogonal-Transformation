import torch
import math
def U_12(a):
    '''
    Compute 2d-Rotation matrices U
    :param a: ndarray (N,) angles a in [0, 2*pi] or a  of angles according to Proposition C.1
    :return: 2x2 Rotation matrix corresponding to a or an array of rotation matrices of size 2x2
    '''
    cos_ = a.cos()
    sin_ = a.sin()
    obe = torch.stack((cos_, sin_), dim=1)
    unte = torch.stack((-sin_, cos_), dim=1)
    return torch.stack((obe, unte), dim=2)

def generate_indices(d):
    '''
    Generates indices of ndarray angle of size (N, p) according to Proposition C.1
    where p=d(d-1)/2
    :param d: dimension
    :return: list of indices
    '''
    indices = []
    for i in range(d-1):
        for j in range(i, d-1):
            indices.append([d-2-(j-i), d-1-(j-i)])
    return indices

def generate_angles_interval(d):
    '''
    Generates upperbound of the intervals of each component of the d(d-1)/2-dimensional angle
    according to Proposition C.1
    :param d: dimension
    :return: list of upperbond of length d(d-1)/2
    '''
    angles = []
    for i in range(d - 1):
        for j in range(i, d - 1):
            if j - i == 0:
                angles.append(2 * math.pi)
            else:
                angles.append(math.pi)
    return angles
def compute_rotation_U(d, alpha):
    '''
    :param d: dimension
    :param alpha: ndarray of angle of size (N, p), where p=d(d-1)/2
    :return: ndarray (N, d, d) of rotation matrices
    '''
    I = generate_indices(d)
    U = torch.eye(d).reshape(1, d, d)
    U = torch.tile(U, (alpha.shape[0], 1, 1))
    for ik, k in enumerate(I):
        U[:, :, k[0]:k[1] + 1] = torch.matmul(U[:, :, k[0]:k[1] + 1], U_12(alpha[:, ik]))
    return U
