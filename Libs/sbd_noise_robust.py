import torch
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def comp_s(matrix):
    '''
    Computes S=\sum_k P^kT P^k
    :param matrix:
    :return:
    '''
    dim = matrix.shape[1]
    kron_l = []
    for i in range(matrix.shape[0]):
        a = matrix[i, :, :]
        b = a.clone().t().contiguous()
        fi = torch.kron(torch.eye(dim), a)
        se = torch.kron(b, torch.eye(dim))
        kron_l.append((fi - se).cpu())
    kron = torch.stack(kron_l, dim=0)
    return torch.sum(kron.permute(0, 2, 1) @ kron, dim=0).to(device)


def get_U(matrix, epsilon1=1e-5, epsilon2=1e-5):
    '''
    Computes P
    :param matrix:
    :param epsilon1:
    :param epsilon2:
    :return:
    '''
    s = comp_s(matrix)
    dim = matrix.shape[1]
    u, d, v = torch.svd(s)
    i = 0
    bed = True
    while bed == True and i < d.shape[0]:
        if d[i] < epsilon1:
            vectors = u.t()[i:]
            bed = False
        i += 1
    sc = torch.rand(vectors.shape[0]) - 0.5
    sc /= torch.norm(sc)
    vectors = vectors * sc.unsqueeze(1)
    U = vectors.sum(dim=0).reshape(dim, dim)
    P, d, V = torch.svd(0.5 * (U + U.t()))
    BlockSizes = []
    start = d[0]
    b = 1
    for i in range(1, dim):
        if i < dim - 1:
            if (start - d[i]) <= epsilon2:
                b += 1
            else:
                if (start - d[i]) > epsilon2:
                    BlockSizes.append(b)
                    b = 1
                    start = d[i]
        else:
            if (start - d[i]) <= epsilon2:
                b += 1
            else:
                BlockSizes.append(b)
                b = 1
            BlockSizes.append(b)
    return P.t(), BlockSizes

