import json
import copy
import torch
from Utils.Evaluation_utils import get_total_Rot, Init_Method

def plot_histogram(data, grad_=False, j=str(0), clamp=1e-4, cov=None):
    import matplotlib.pyplot as plt
    import numpy as np
    print(data)
    end = 7 if j == str(0) else 8
    key = '1' if grad_ else '2'
    norm = '\infty'#'1'
    if len(data[key]['labels']) > 0:
        name_partial = "Gradient" if grad_ else "Hessian"
        name1 = 'grad' if grad_ else 'hessian'
        norm_key = '1-norm' if norm == '1' else 'inf-norm'
        p_indices = np.argsort(np.array(data[key]['recons'][norm_key]))
        sort_param = p_indices[::-1][:end]
        print('\n p_indices: ', p_indices)
        couplings = tuple([set([int(j)+1 for j in data[key]['labels'][i]]) for i in sort_param])
        print('\n couplings: ', couplings)
        values = data[key][name1][norm_key] + data[key]['recons'][norm_key]
        evaluations = {
            "Anova-term": tuple([data[key]['recons'][norm_key][i] for i in sort_param]),
            name_partial: tuple([data[key][name1][norm_key][i] for i in sort_param]),
        }
        x = np.arange(len(couplings))
        width = 0.2
        multiplier = 0
        fig, ax = plt.subplots(layout='constrained')
        for attribute, measurement in evaluations.items():
            offset = width * multiplier
            print(attribute)
            if attribute == 'Gradient' or attribute =='Hessian':
                if norm == "\infty":
                    label_ = r"$||\partial_{\mathbf{v}}f||_{\infty}$"
                else:
                    label_ = r"$||\partial_{\mathbf{v}}f||_1$"
                rects = ax.bar(x + offset, measurement, width, label=label_)
            else:
                if norm == "\infty":
                    label_ = r"$||f_{\mathbf{v}_{\max}, A}||_{\infty}$"
                else:
                    label_ = r"$||f_{\mathbf{v}_{\max}, A}||_1$"
                rects = ax.bar(x + offset, measurement, width, label=label_)
            ax.legend(ncol=1)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.axhline(y=clamp, color='r', linestyle='dashed')

        if cov is not None:
            ax.set_xlabel('Couplings $\mathbf{v}$')
        ax.set_xticks(x + width, couplings)
        ax.legend(loc='upper right', ncols=3)
        ax.set_yscale('log')
        ax.set_ylim(0, 3*max(values)/2)
        plt.show()

if __name__ == '__main__':
    output_folder = '/homes/numerik/fatimaba/store/Github/trafo_nova/Anova_AE/Output_files'
    cov = None
    init_method = Init_Method.GS
    h_sizes = [1]
    for h_size in h_sizes:
        print("\n .................................h_size: ", h_size)
        suff_ = '_cov_{}'.format(cov) if cov is not None else ''
        if init_method == Init_Method.GS:
            name = '{}/Test_anova{}_h_size_{}.json'.format(output_folder, suff_, h_size)
        else:
            name = '{}/Test_anova{}_mo.json'.format(output_folder, suff_)
        with open(name) as convert_file:
            datas = copy.deepcopy(json.load(convert_file))
            for j in datas.keys():
                dim = datas[str(j)]['dim']
                Rot_la, Rot_re = get_total_Rot(datas, str(j), init_method=init_method)
                hessian_supp = torch.as_tensor(datas['0']['svd_basis'])
                supp = hessian_supp.shape[1]
                N = hessian_supp.shape[0]
                x_test = torch.as_tensor(datas[str(j)]['x_test'])
                ground_truth = datas[str(j)]['groundtruth']
                ground_truth['v'] = torch.as_tensor(ground_truth['v'])
                U = torch.as_tensor(datas[str(j)]['grad_U'])
                Rot = Rot_re.T
                suff_ = '_cov_{}'.format(cov) if cov is not None else ''
                fname = ''
                if init_method == Init_Method.GS:
                    fname = '{}/Anova_deriv{}_h_size_{}_{}.json'.format(output_folder, suff_, h_size, j)
                    with open(fname) as convert_file:
                        results_ = copy.deepcopy(json.load(convert_file))
                else:
                    fname = '{}/Anova_deriv{}_mo_{}.json'.format(output_folder, suff_, j)
                    with open(fname) as convert_file:
                        results_ = copy.deepcopy(json.load(convert_file))
                print('\n Filename: ', fname)
                plot_histogram(results_, j=str(j), cov=cov)