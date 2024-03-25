import torch
import math
def rot_12(a):
    '''
    Compute 2d-Rotation matrices

    :param a: angle a \in R or a 2d-array of angles
    :return: 2x2 Rotation matrix corresponding to a or an array of rotation matrices of size 2x2
    '''
    cos_ = a.cos()
    sin_ = a.sin()
    obe = torch.stack((cos_, sin_), dim=1)
    unte = torch.stack((-sin_, cos_), dim=1)
    return torch.stack((obe, unte), dim=2)

def generate_indices(d):
    '''
    Generate indices of
    :param d: dimension
    :return: indices
    '''
    indices = []
    for i in range(d-1):
        for j in range(i, d-1):
            indices.append([d-2-(j-i), d-1-(j-i)])
    return indices

def generate_angles_interval(d):
    '''

    :param d:
    :return:
    '''
    angles = []
    for i in range(d - 1):
        for j in range(i, d - 1):
            if j - i == 0:
                angles.append(2 * math.pi)
            else:
                angles.append(math.pi)
    return angles
def compute_rot(d, alpha):
    '''
    :param d:
    :param alpha:
    :return:
    '''
    I = generate_indices(d)
    Rot = torch.eye(d).reshape(1, d, d)
    Rot = torch.tile(Rot, (alpha.shape[0], 1, 1))
    for ik, k in enumerate(I):
        Rot[:, :, k[0]:k[1] + 1] = torch.matmul(Rot[:, :, k[0]:k[1] + 1], rot_12(alpha[:, ik]))
    return Rot
