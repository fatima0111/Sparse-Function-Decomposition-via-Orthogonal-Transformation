import json
import copy
import torch
from Utils.Evaluation_utils import get_total_Rot, Init_Method

def plot_histogram(data,j=str(0), clamp=1e-4, cov=None):
    import matplotlib.pyplot as plt
    import numpy as np
    print(data)
    end = 7 if j == str(0) else 8
    p = ['inf', '1']
    for key in data.keys():
        for norm in p:
            norm_key = '1-norm' if norm == str(1) else 'inf-norm'
            if len(data[key]['labels']) > 0:
                name1 = 'grad' if key == str(1) else 'hessian'
                name_partial = 'Gradient' if key == str(1) else 'Hessian'
                p_indices = np.argsort(np.array(data[key]['recons'][norm_key]))
                sort_param = p_indices[::-1][:end]
                part_array = np.array(data[key][name1][norm_key])
                const = 6 if key == str(1) else 5
                max_gradient = (2 ** const) * np.max(part_array[np.where(part_array <= clamp)[0]])
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
                    if attribute == 'Gradient':
                        label_ = r"$||\partial_{\{i\}}f||_{\infty}$" if norm == "inf" else r"$||\partial_{\{i\}}f||_1$"
                    elif attribute == 'Hessian':
                        label_ = r"$||\partial_{\{i, j\}}f||_{\infty}$" if norm == "inf" else r"$||\partial_{\{i, j\}}f||_1$"
                    else:
                        if key == str(1):
                            label_ = r"$||f||_{\{i\}, \infty}$" if norm == "inf" else r"$||f||_{\{i\}, 1}$"
                        else:
                            label_ = r"$||f||_{\{i, j\}, \infty}$" if norm == "inf" else r"$||f||_{\{i, j\}, 1}$"
                    rects = ax.bar(x + offset, measurement, width, label=label_)
                    ax.legend(ncol=1)
                    multiplier += 1
                ax.axhline(y=clamp, color='r', linestyle='dashed')
                ax.axhline(y=max_gradient, color='green', linestyle='dashed')
                if cov is not None:
                    ax.set_xlabel('Couplings')
                ax.set_xticks(x + width, couplings)
                ax.legend(loc='upper right', ncols=3)
                ax.set_yscale('log')
                ax.set_ylim(0, 3*max(values)/2)
                suff = '_cov_{}'.format(cov) if cov is not None else ''
                plt.savefig('Plots/Anova_terms/{}d_couplings_f{}_L{}{}.png'.format(key, j, norm, suff))
    plt.show()

if __name__ == '__main__':
    output_folder = '/homes/numerik/fatimaba/store/Github/trafo_nova/Anova_AE/Output_files'
    covs = [None, 0.5]
    init_method = Init_Method.GS
    h_sizes = [1]
    for h_size in h_sizes:
        print("\n .................................h_size: ", h_size)
        for cov in covs:
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
                    x_test = torch.as_tensor(datas[str(j)]['x_test'])
                    ground_truth = datas[str(j)]['groundtruth']
                    ground_truth['R'] = torch.as_tensor(ground_truth['R'])
                    #U_v = torch.as_tensor(datas[str(j)]['U_v'])
                    #Rot = Rot_re.T
                    suff_ = '_cov_{}'.format(cov) if cov is not None else ''
                    fname = ''
                    if init_method == Init_Method.GS:
                        fname = '{}/ANOVA_deriv{}_h_size_{}_{}.json'.format(output_folder, suff_, h_size, j)
                        with open(fname) as convert_file:
                            results_ = copy.deepcopy(json.load(convert_file))
                    else:
                        fname = '{}/ANOVA_deriv{}_mo_{}.json'.format(output_folder, suff_, j)
                        with open(fname) as convert_file:
                            results_ = copy.deepcopy(json.load(convert_file))
                    print('\n Filename: ', fname)
                    plot_histogram(results_, j=str(j), cov=cov)