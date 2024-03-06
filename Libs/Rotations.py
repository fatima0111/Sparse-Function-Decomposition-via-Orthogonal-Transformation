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
def compute_rotation_matrix_right(d, alpha):
    '''
    :param d:
    :param alpha:
    :return:
    '''
    R = torch.eye(d).reshape(1, d, d)
    R_right = torch.tile(R, (alpha.shape[0], 1, 1))
    R_right[:, :d - 1, :d - 1] = compute_rotation_matrix(d - 1, alpha)
    return R_right

def compute_rotation_matrix(d, alpha):
    '''
    Compute rotation matrix from parameters
    :param n: Space dimension
    :param parameters:
    :return:
    '''
    if d <= 1 or d > 5:
        exit(1)
    elif d == 2:
        return rot_12(alpha.squeeze())
    else:
        R_left = compute_rotation_matrix_left(d, alpha[:, :d - 1])
        R_right = compute_rotation_matrix_right(d, alpha[:, d - 1:])
        R= R_left @ R_right
        return R
def compute_rotation_matrix_left(d, alpha):
    '''

    :param d:
    :param alpha:
    :return:
    '''
    R_left = torch.eye(d).reshape(1, d, d)
    R_left = torch.tile(R_left, (alpha.shape[0], 1, 1))
    for k in range(d - 1):
        R_left[:, :, k:k + 2] = torch.matmul(R_left[:, :, k:k+2], rot_12(alpha[:, k]))
    return R_left

def generate_indices(d):
    indices = []
    for i in range(d-1):
        for j in range(i, d-1):
            indices.append([d-2-(j-i), d-1-(j-i)])
    return indices

def generate_angles_interval(d):
    angles = []
    for i in range(d - 1):
        for j in range(i, d - 1):
            if j - i == 0:
                angles.append(2 * math.pi)
            else:
                angles.append(math.pi)
    return angles
def compute_rot(d, alpha):
    I = generate_indices(d)
    Rot = torch.eye(d).reshape(1, d, d)
    Rot = torch.tile(Rot, (alpha.shape[0], 1, 1))
    for ik, k in enumerate(I):
        Rot[:, :, k[0]:k[1] + 1] = torch.matmul(Rot[:, :, k[0]:k[1] + 1], rot_12(alpha[:, ik]))
    return Rot
