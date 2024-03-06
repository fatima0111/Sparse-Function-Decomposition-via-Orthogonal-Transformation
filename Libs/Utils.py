import torch
import numpy as np
from Anova_NN.Model_class import SparseAdditiveModel
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.autograd.functional import jacobian, hessian
import itertools
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

possible_sub_functions = [lambda x, p: x ** p,
                          lambda x, p: torch.sin(x * p),
                          lambda x, p: torch.cos(x * p),
                          lambda x, p: (x ** 2 + p ** 2)**(1/3),
                          lambda x, p: torch.exp(-(x-p)**2),
                          lambda x, p: x + p
                          ]

first_order_derivate = [lambda x, p: p*(x**(p-1)),
                              lambda x, p: p*torch.cos(x*p),
                              lambda x, p: -p*torch.sin(x * p),
                              lambda x, p: (2*x)/(3*(x**2 +p**2)**(2/3)),
                              lambda x, p: -2*torch.exp(-(x-p)**2)*(x-p),
                              lambda x, p: torch.ones(x.shape[0])
                          ]
second_order_derivative = [lambda x, p: (p-1)*p*(x**(p-2)) if p-2>=0 else p*torch.zeros(x.shape[0]),
                           lambda x, p: -(p**2)*(torch.sin(x*p)),
                           lambda x, p: -(p**2)*torch.cos(x * p),
                           lambda x, p: -2*(x**2-3*p**2)/(9*(x**2 +p**2)**(5/3)),
                           lambda x, p: torch.exp(-(x-p)**2)*(4*(x**2)-8*p*x+4*(p**2)-2),
                           lambda x, p: torch.zeros(x.shape[0])

]


def random_function(x, ground_truth, n_A=None, A_=None, y=None, fi=None,
                    ind_i=None):
    """
    For computing the i-th anova_function just set num_sub_functions=[i]
    :param input_data:
    :param num_sub_functions: list of involved anova functions can be range of N if N functions are involved
    :param num_sub_function_entries:
    :param sub_function_indices:
    :param entry_indices:
    :param parameters:
    :return:
    """
    if x.ndim == 1:
        x = x.unsqueeze(dim=0)
    num_sub_functions = ground_truth['K']
    A = ground_truth['v'] if A_ is None and n_A is None else ground_truth[
        'v'][n_A] if A_ is None and n_A is not None else A_
    entry_indices = ground_truth['U']
    parameters = ground_truth['parameters']
    num_sub_function_entries = ground_truth['t_num_sub_function_entries']
    sub_function_indices = ground_truth['t_sub_function_indices']
    #print(A.dtype, x.dtype)
    input_data = torch.matmul(A, x.T).T if y is None else torch.matmul(A, y.T).T
    # the sub-functions are multiplied, i.e. f_i(x_1, x_2, x_3) = f_k1(x_1) * f_k2(x_2) * f_k3(x_2)
    num_samples = input_data.shape[0]
    out_f = torch.zeros(num_samples)
    coeff = ground_truth['coeff'] if 'coeff' in ground_truth.keys() else torch.ones(num_sub_functions)
    if type(num_sub_functions) == int:
        num_sub_functions = list(range(num_sub_functions))
        #print(num_sub_functions, ground_truth)
    if len(num_sub_functions) == 1 and fi is not None:
        if type(num_sub_functions) != int and ind_i is not None:
            if fi == 5:
                d = 10
            elif fi == 6 or fi == 7:
                d = 4
            elif fi == 8:
                d = 9
            lambda_ = np.random.uniform(size=d-1)
            X = np.ones((input_data.shape[0], d))
            indices = list(range(d))
            print(indices, ind_i)
            indices.pop(ind_i)
            print(indices, ind_i)
            X[:, indices] = lambda_*torch.ones((input_data.shape[0], d-1))
            print( X[:, ind_i].shape, input_data.shape)
            X[:, ind_i] = input_data
            input_data = X
        out_f = possible_sub_functions[fi](input_data, 0)
    else:
        for i in num_sub_functions:
            f_part = torch.ones(num_samples)
            for k in range(num_sub_function_entries[i]):
                # get a random extra parameter for the sub function
                # parameter = np.random.randint(0, 3) + 1
                parameter = parameters[i][k]
                f_part *= possible_sub_functions[sub_function_indices[i][k]](
                    input_data[:, entry_indices[i][k]], parameter)
            out_f += coeff[i]*f_part
    return out_f


def proj_orth(Q):
    u, d, v_h = torch.linalg.svd(Q)
    Q = torch.matmul(u, v_h)
    return Q




def train_model(X, y, K, Optimizer=torch.optim.Adam, num_epochs=int(1e4),
                       learning_rate=5e-4, batch_size=5000,
                       fi=None, my_print='', sigma=1, lambd=0, pen=1):
    num_samples, data_dimension = X.shape
    num_samples //= 2

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(y_train.shape, y_test.shape)
    #sigma = ((y_train.squeeze() ** 2).mean() - (y_train.squeeze().mean()) ** 2) if sigma is None else sigma #if fi is not None else 1
    y_train /= sigma

    n_train = X_train.shape[0]
    num_subnets = K
    model = SparseAdditiveModel(data_dimension, num_subnets, setting=1)
    model.set_sigma(sigma)
    optimizer = Optimizer(model.parameters(),
                          lr=learning_rate)#,weight_decay=1e-8

    lmbda = lambda epoch: .85
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)
    test_min = 1e+10
    train_min = 1e+10

    params_min = model.state_dict()
    for j in range(num_epochs): #tqdm(range(num_epochs)):
        accumulated_loss = 0
        fidelity_loss = 0
        random_batch_indices = np.random.permutation(np.arange(n_train))
        for i in range(int(np.ceil(n_train / batch_size))):
            batch_indices = random_batch_indices[i * batch_size: (i + 1) * batch_size]
            input_batch = X_train[batch_indices]
            target_batch = y_train[batch_indices]
            model_out = model(input_batch, mode=False).squeeze()
            #print(model_out.shape, target_batch.squeeze().shape)
            data_fidelity_loss = ((model_out - target_batch.squeeze()) ** 2).mean()

            #reg_term = model.get_penalty() if pen==1 else model.get_penalty_jac(X_test)
            loss = data_fidelity_loss #+ lambd*reg_term
            optimizer.zero_grad()
            #print(i, j)
            loss.backward()
            optimizer.step()

            accumulated_loss += loss.data.item() / int(np.ceil(n_train / batch_size))
            fidelity_loss += data_fidelity_loss.data.item() / int(np.ceil(n_train / batch_size))
            with torch.no_grad():
                test_model_out = model(X_test, mode=True).squeeze()
                test_fidelity_loss = ((test_model_out - y_test.squeeze()) ** 2).mean()
                test_loss = test_fidelity_loss
                if test_loss < test_min: #and j > 10:
                    test_min = test_loss.clone()
                    train_min = data_fidelity_loss
                    params_min = model.state_dict()
            if j % 100 == 0:
                print('training_data_fidelity: {} test_data_fidelity_loss {}'.format(fidelity_loss,
                                                                                     test_fidelity_loss))

                my_print += '\n training_data_fidelity: {} test_data_fidelity_loss {} \n'.format(fidelity_loss,
                                                                                     test_fidelity_loss)
            if j % 550 == 0 and scheduler.get_last_lr()[0] > 1e-5:
                scheduler.step()
                print('***** Learning_rate: {}'.format(scheduler.get_last_lr()[0]))
                my_print += '\n ***** Learning_rate: {}\n'.format(scheduler.get_last_lr()[0])

                print("\n train_min {} test_min : {}\n".format(train_min, test_min))
                my_print += "\n train_min {} test_min : {}\n".format(train_min, test_min)
    return params_min, my_print


def jacobian2(y, x, create_graph=False):
    '''

    :param y:
    :param x:
    :param create_graph:
    :return:
    '''
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True,
                                      create_graph=create_graph, allow_unused=True)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian2(y, x):
    '''

    :param y:
    :param x:
    :return:
    '''
    z = jacobian2(y, x, create_graph=True)
    return jacobian2(z, x)


def compute_hessian(x, model=None):
    '''

    :param x:
    :param model:
    :return:
    '''
    num_samples, data_dimension = x.shape
    def f(x):
        out = model(x)
        return out
    hessian_ = torch.zeros((num_samples, data_dimension, data_dimension))
    for i in range(num_samples):
        x_i = x[i, :]
        x_i = x_i
        x_i.requires_grad = True

        val = hessian(model, x_i)
        hessian_[i, :, :] = val.squeeze()
    return hessian_

def compute_hessian_auto(x, ground_truth=None, model=None):
    '''

    :param x:
    :param ground_truth:
    :param model:
    :return:
    '''
    num_samples, data_dimension = x.shape
    from torch.autograd.functional import hessian
    def f(ground_truth):
        def get_random(x):
            if ground_truth is not None:
                return random_function(x,
                                       ground_truth=ground_truth)
            else:
                return model(x)
        return get_random
    hessian_ = torch.zeros((num_samples, data_dimension, data_dimension))
    for i in range(num_samples):
        x_i = x[i, :]
        x_i = x_i
        x_i.requires_grad = True
        val = hessian(f(ground_truth), x_i)  # hessian2(model(x_i),xi)
        hessian_[i, :, :] = val.squeeze()
    return hessian_  # .cpu().detach().to(device)

def compute_gradient_autograd(x, ground_truth=None, model=None):
    '''

    :param x:
    :param ground_truth:
    :param model:
    :return:
    '''
    def f(ground_truth):
        def get_random(x):
            if ground_truth is not None:
                return random_function(x,ground_truth=ground_truth)
            else:
                return model(x)
        return get_random
    #x.requires_grad = True
    gradient = torch.zeros_like(x.T)
    print(x.shape)
    for i in range(x.shape[0]):
        x_i = x[i:i+1, :]
        x_i.required_grad = True
        gradient[:, i] = torch.autograd.functional.jacobian(f(ground_truth), x_i, create_graph=True)
        #print(gradient.shape)
    return gradient.detach()


def get_active_attribute_min_value(U, svd_value):
    '''

    :param U:
    :param svd_value:
    :return:
    '''
    active_variables = list(itertools.chain(*U))
    active_variables.sort()
    active_variables = list(set(active_variables))
    num_active_attribute = len(active_variables)
    #sorted_svd_value = svd_value.sort()
    min_value_active_attribute = svd_value[num_active_attribute-1]
    print("\n min_value_active_attribute: ", min_value_active_attribute)
    return min_value_active_attribute


def compute_gradient_orig_2d(x, ground_truth, n_A=None,
                             return_coupling=False):
    '''

    :param x:
    :param ground_truth:
    :param n_A:
    :param return_coupling:
    :return:
    '''
    n = x.shape[1]
    U = ground_truth['U']
    K = ground_truth['K']
    A = ground_truth['v'][n_A] if n_A is not None else ground_truth['v']
    y = torch.matmul(A, x.T).T
    #print(y.shape, x.shape)
    gradient_f = torch.zeros([x.shape[1], x.shape[0]])
    t_sub_function_indices = ground_truth['t_sub_function_indices']
    parameters = ground_truth['parameters']
    active_variables = list(itertools.chain(*U))
    active_variables.sort()
    active_variables = list(set(active_variables))
    coeff = ground_truth['coeff'] if 'coeff' in ground_truth.keys() else torch.ones(K)
    for i in range(n):
        gradient_i = torch.zeros(x.shape[0])
        if i in active_variables:
            for k in range(K):
                ind_u = U.index(U[k])
                f_k = t_sub_function_indices[ind_u]
                p_k = parameters[ind_u]
                if i in U[k]:
                    ind_i = U[k].index(i)
                    if U[k][0] == U[k][1]:
                        gradient_i += coeff[k]*first_order_derivate[f_k[ind_i]](
                            y[:, i], p_k[ind_i])
                    elif U[k][0] != U[k][1]:
                        gradient_i += coeff[k]*first_order_derivate[f_k[ind_i]](
                            y[:, i], p_k[ind_i])*possible_sub_functions[
                            f_k[(ind_i+1) % 2]](
                            y[:, U[k][(ind_i+1) % 2]], p_k[(ind_i+1) % 2]
                        )
            gradient_f[i, :] = gradient_i
    #print(gradient_f.abs().mean(dim=1))
    if return_coupling:
        return gradient_f
    gradient = torch.matmul(A.T, gradient_f)
    return gradient


def compute_hessian_orig_2d(x, ground_truth, return_coupling=False,
                            n_A=None):
    '''

    :param x:
    :param ground_truth:
    :param return_coupling:
    :param n_A:
    :return:
    '''
    n = x.shape[1]
    U = ground_truth['U']
    K = ground_truth['K']
    A = ground_truth['v'][n_A] if n_A is not None else ground_truth['v']
    y = torch.matmul(A, x.T).T
    hessian_f = torch.zeros([x.shape[0], x.shape[1], x.shape[1]])
    t_sub_function_indices = ground_truth['t_sub_function_indices']
    parameters = ground_truth['parameters']
    active_variables = list(itertools.chain(*U))
    active_variables.sort()
    active_variables = list(set(active_variables))
    coeff = ground_truth['coeff'] if 'coeff' in ground_truth.keys() else torch.ones(K)
    #print('\n coeff: ', coeff)
    for i in range(n):
        for j in range(n):
            hessian_ij = torch.zeros(x.shape[0])
            if i in active_variables and j in active_variables:
                for k in range(K):
                    ind_u = U.index(U[k])
                    if i in U[k] and j in U[k]:
                        ind_i = U[k].index(i)
                        ind_j = U[k].index(j)
                    f_k = t_sub_function_indices[ind_u]
                    p_k = parameters[ind_u]
                    if i == j and i in U[k]:
                        if U[k][0] == U[k][1]:
                            hessian_ij += coeff[k]*second_order_derivative[f_k[ind_i]](x[:, i], p_k[ind_i])
                        elif U[k][0] != U[k][1]:
                            hessian_ij += coeff[k]*second_order_derivative[f_k[ind_i]](
                                y[:, i], p_k[ind_i])*possible_sub_functions[
                                f_k[(ind_i+1) % 2]](
                                y[:, U[k][(ind_i+1) % 2]], p_k[(ind_i+1) % 2]
                            )
                    elif i != j and i in U[k] and j in U[k]:
                        #print("f_k[ind_u]: ", f_k, f_k[ind_i], f_k[ind_j], i, j)
                        hessian_ij += coeff[k]*first_order_derivate[f_k[ind_i]](
                            y[:, i], p_k[ind_i])*first_order_derivate[f_k[ind_j]](
                            y[:, j], p_k[ind_j])

            hessian_f[:, i, j] = hessian_ij
    if return_coupling:
        hessian_orig = hessian_f.abs().mean(dim=0)
        hessian_orig[hessian_orig != torch.clamp(hessian_orig, 1e-4)] = 0
        C = torch.where(hessian_orig > 0)
        return hessian_f, [[C[0][i].item(), C[1][i].item()] for i in range(len(C[0]))]
    hessian = torch.matmul(torch.matmul(A.T, hessian_f), A)
    return hessian

def model_backward(x, model=None, f=None):
    x.requires_grad = True
    loss = model(x).squeeze() if model is None else f(x)
    loss.backward(torch.ones(loss.shape))
    return x.grad.detach().T

def noise_function(sample, means_=None, cov=.5, type='0', dtype= torch.float64):
    '''

    :param sample:
    :param means:
    :param cov:
    :param type: 0=Gradient pdf, 1=Hessian pdf, else:pdf
    :return:
    '''
    d = sample.shape[1]
    cs = {0.5:2000,
          1:7500,
          1.5:60000,
          2:10000
}
    denom = 1 / cs[cov]
    cov_mat = cov * torch.eye(d)
    den = -1 / (2 * cov)
    if means_ is None:
        means_ = [-.5, 1.5]
    means_list = []
    for i in range(d):
        means_list.append(means_)
    means = list(itertools.product(*means_list))
    exp = torch.zeros(len(means), sample.shape[0])
    mean_v = torch.as_tensor(means)
    for ind_mean, mean in enumerate(means):
        mean = torch.as_tensor(mean)
        diff = sample - torch.as_tensor(mean)
        exp[ind_mean, :] = torch.exp(-((diff @ torch.linalg.inv(cov_mat)) * diff).sum(dim=1) / 2)
    if type == '0': #Gradient
        gradient = torch.zeros_like(sample.T,  dtype=dtype)
        for i in range(d):
            mean_i = mean_v[:, i]
            diff_i = mean_i - torch.tile(sample[:, i], (mean_i.shape[0], 1)).T
            gradient[i, :] = (den*diff_i.T*exp).sum(dim=0)
        return denom*gradient
    elif type == '1': #Hessian
        hessian = torch.zeros((sample.shape[0], d, d))
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    mean_i = mean_v[:, i]
                    diff_i = mean_i-torch.tile(sample[:, i], (mean_i.shape[0], 1)).T
                    hessian_i = (den*(den*diff_i.T**2 +1)*exp).sum(dim=0)
                else:
                    diff_i = mean_v[:, i] - torch.tile(sample[:, i], (mean_v[:, i].shape[0], 1)).T
                    diff_j = mean_v[:, j] - torch.tile(sample[:, j], (mean_v[:, j].shape[0], 1)).T
                    hessian_i = (den**2 * diff_i.T*diff_j.T*exp).sum(dim=0)
                hessian[:, i, j] = hessian[:, j, i] = hessian_i
        return hessian*denom
    else:
        return exp.sum(dim=0)*denom

