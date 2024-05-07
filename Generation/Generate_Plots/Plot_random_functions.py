import json
import numpy as np
from matplotlib import pyplot as plt
from Utils.Function_utils import compute_hessian_orig_2d
from Utils.Evaluation_utils import get_total_Rot, Init_Method, loss_function12
import torch
from os.path import dirname, abspath

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    #device = torch.device('cuda')
#else:
#    if torch.cuda.is_available() else 'cpu'
gtype = torch.float64

out_dir = dirname(dirname(abspath(__file__)))+'/Output_algorithms/Random_functions'
report_dir = dirname(dirname(abspath(__file__))) + '/Plots/Random_functions'

names_clean = [
    'function_Man_opt_grid_search_h_1.0_M_50.json',
    'function_Man_opt_grid_search_h_0.5_M_50.json',
    'function_Man_opt_grid_search_h_0.25_M_50.json',
    'function_Man_opt_grid_search_h_0.125_M_50.json'
]
names_noisy = [
    'function_Man_opt_grid_search_cov_0.5_h_1.0_M_50.json',
    'function_Man_opt_grid_search_cov_0.5_h_0.5_M_50.json',
    'function_Man_opt_grid_search_cov_0.5_h_0.25_M_50.json',
    'function_Man_opt_grid_search_cov_0.5_h_0.125_M_50.json']
etas_clean = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
clamp_gt = 1e-6
etas_noisy = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5]
len_loss = 70
markers = [('d', 'thin_diamond'), ('o', 'circle'), ('x', 'x'), ('+', 'plus'), ('H', 'hexagon2'), ('h', 'hexagon1'),
            ('*', 'star'), ('P', 'plus (filled)'), ('p', 'pentagon'), ('s', 'square'), ('8', 'octagon'),
            ('4', 'tri_right'), ('3', 'tri_left'), ('2', 'tri_up'), ('1', 'tri_down'), ('>', 'triangle_right'),
            ('<', 'triangle_left'), ('^', 'triangle_up'), ('v', 'triangle_down'), (',', 'pixel'), ('.', 'point'),
            ('D', 'diamond'), ('_', 'hline'), ('X', 'x (filled)'), ('|', 'vline')]
loss_mean = 0
losses_inner = []
h_sizes = [1, 0.5, 0.25, 0.125]
labels = ['h={}'.format(i) for i in h_sizes]
def plot_loss_ratio(etas, names, noisy_data=False):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    man_opt_clamp = torch.zeros(2, len(etas))
    man_opt_gs_clamp = {}
    for h in h_sizes:
        man_opt_gs_clamp[h] = {}
        man_opt_gs_clamp[h]['la'] = torch.zeros(len(etas))
        man_opt_gs_clamp[h]['rgd'] = torch.zeros(len(etas))

    for ind_names, name in enumerate(names):
        with open('{}/{}'.format(out_dir, name), 'r') as convert_file:
            data = json.load(convert_file)
            c_losses_man_opt_la = []
            c_losses_man_opt_rgd = []
            c_losses_man_opt_gs_la = []
            c_losses_man_opt_gs_rgd = []
            c_loss_mean_man_opt = torch.zeros(2, len_loss)
            c_loss_mean_man_opt_gs = torch.zeros(2, len_loss)
            print('{}/{}'.format(out_dir, name))
            losses = []
            losses_mo = []
            for ind_clamp, clamp in enumerate(etas):
                x_0 = 0
                for j in data.keys():
                    if len(data[j]['blocs_alg']) > 1:
                        x_test = data[j]['x_test'] = torch.as_tensor(data[j]['x_test'], dtype=gtype)
                        supp = int(data[j]['rank_grad'])
                        d = (data[j]['groundtruth']['d'])
                        h_size = float(name.split('_h_')[1].split('_M')[0])
                        ground_truth = data[j]['groundtruth']
                        ground_truth['R'] = torch.as_tensor(ground_truth['R'], dtype=gtype)
                        R = ground_truth['R']
                        rank = int(data[j]['rank_hessian'])
                        #print('\n data: ', j, clamp)
                        hessianF_orig = compute_hessian_orig_2d(x_test.clone(), ground_truth)
                        hessian_f, _ = compute_hessian_orig_2d(x_test.clone(), ground_truth, return_coupling=True)
                        if ind_clamp == len(etas)-1:
                            u_grad = torch.as_tensor(data[j]['U_svd'], dtype=gtype)
                            u2, d2, v2 = torch.svd((u_grad.T @ hessianF_orig @ u_grad).flatten(start_dim=1).T)
                            hessian_basis_1 = u2.T[:rank].reshape(rank, d, d)
                            hessian_basis = u_grad @ hessian_basis_1 @ u_grad.T
                            loss = loss_function12(R, hessian_basis)
                            blocSizes = data[j]['blocs_alg']
                            hessian_rank = torch.as_tensor(data[j]['hessian_basis'], dtype=gtype)
                            P = torch.as_tensor(data[j]['U_bloc'], dtype=gtype)
                            hessian_rank_blocks = P @ hessian_rank @ P.T

                        H_gt = ((R @ hessianF_orig @ R.T).abs()).mean(dim=0)
                        H_gt[H_gt != torch.clamp(H_gt, 1e-5)] = 0
                        zero_norm_ht = len(H_gt.nonzero())
                        Rot_la, Rot_rgd = get_total_Rot(data, j, Init_Method.GS)
                        H_la = ((Rot_la@hessianF_orig@Rot_la.T).abs()).mean(dim=0)
                        H_la[H_la != torch.clamp(H_la, clamp)] = 0
                        zero_norm_H_la = len(H_la.nonzero())

                        H_rgd = ((Rot_rgd@hessianF_orig@Rot_rgd.T).abs()).mean(dim=0)
                        H_rgd[H_rgd != torch.clamp(H_rgd, clamp)] = 0
                        zero_norm_H_rgd = len(H_rgd.nonzero())
                        if zero_norm_H_la - zero_norm_ht > 0:
                            man_opt_gs_clamp[h_size]['la'][ind_clamp] += 1
                        if zero_norm_H_rgd - zero_norm_ht > 0:
                            man_opt_gs_clamp[h_size]['rgd'][ind_clamp] += 1
                        if h_size == 1 / 2:
                            Rot_mo_la, Rot_mo_rgd = get_total_Rot(data, j)
                            H_mo_la = ((Rot_mo_la @ hessianF_orig @ Rot_mo_la.T).abs()).mean(dim=0)
                            H_mo_la[H_mo_la != torch.clamp(H_mo_la, clamp)] = 0
                            zero_norm_H_mo_la = len(H_mo_la.nonzero())

                            H_mo_rgd = ((Rot_mo_rgd @ hessianF_orig @ Rot_mo_rgd.T).abs()).mean(dim=0)
                            H_mo_rgd[H_mo_rgd != torch.clamp(H_mo_rgd, clamp)] = 0
                            zero_norm_H_mo_rgd = len(H_mo_rgd.nonzero())
                            if zero_norm_H_mo_la - zero_norm_ht > 0:
                                man_opt_clamp[0, ind_clamp] += 1
                            if zero_norm_H_mo_rgd - zero_norm_ht > 0:
                                    man_opt_clamp[1, ind_clamp] += 1
                            if ind_clamp == len(etas) - 1:
                                man_opt_loss_la_ = torch.zeros(len_loss)
                                man_opt_loss_rgd_ = torch.zeros(len_loss)
                                b_ad = 0
                                ind_b = '0'
                                for b in blocSizes:
                                    if b > 1 and b <= 4:
                                        man_opt_loss_la = data[j]['loss']['Man_Opt_RI'][ind_b]['la']
                                        man_opt_loss_la = man_opt_loss_la[:min(len_loss + 1, len(man_opt_loss_la))]
                                        man_opt_loss_la += [man_opt_loss_la[-1]] * (len_loss - len(man_opt_loss_la))

                                        man_opt_loss_rgd = data[j]['loss']['Man_Opt_RI'][ind_b]['rgd']
                                        man_opt_loss_rgd = man_opt_loss_rgd[:min(len_loss + 1, len(man_opt_loss_rgd))]
                                        man_opt_loss_rgd += [man_opt_loss_rgd[-1]] * (len_loss - len(man_opt_loss_rgd))

                                        man_opt_loss_la_ += torch.as_tensor(man_opt_loss_la, dtype=gtype)
                                        man_opt_loss_rgd_ += torch.as_tensor(man_opt_loss_rgd, dtype=gtype)

                                        ind_b = int(ind_b) + 1
                                        ind_b = str(ind_b)
                                    else:
                                        hessian_rank_b = hessian_rank_blocks[:, b_ad:b_ad + b, b_ad:b_ad + b]
                                        val = ((hessian_rank_b**2).mean(dim=0)).sqrt()
                                        man_opt_loss_la_ += torch.ones(len_loss)*val.squeeze()
                                        man_opt_loss_rgd_ += torch.ones(len_loss)*val.squeeze()
                                    b_ad += b
                                P1 = torch.eye(d, dtype=gtype)
                                P1[:supp, :supp] = P
                                c_losses_man_opt_la.append(man_opt_loss_la_)
                                c_losses_man_opt_rgd.append(man_opt_loss_rgd_)
                                losses_mo.append(loss)

                        if ind_clamp == len(etas) - 1:
                            man_opt_gs_loss_la_ = torch.zeros(len_loss)
                            man_opt_gs_loss_rgd_ = torch.zeros(len_loss)
                            b_ad = 0
                            ind_b = '0'
                            for b in blocSizes:
                                if b > 1 and b <= 4:
                                    man_opt_gs_loss_la = data[j]['loss']['Man_Opt_GS'][ind_b]['la']
                                    man_opt_gs_loss_la = man_opt_gs_loss_la[:min(len_loss + 1, len(man_opt_gs_loss_la))]
                                    man_opt_gs_loss_la += [man_opt_gs_loss_la[-1]] * (max(0, len_loss - len(man_opt_gs_loss_la)))
                                    man_opt_gs_loss_la[0] = data[j]['loss']['Grid_search'][ind_b]

                                    man_opt_gs_loss_rgd = data[j]['loss']['Man_Opt_GS'][ind_b]['rgd']
                                    man_opt_gs_loss_rgd = man_opt_gs_loss_rgd[:min(len_loss + 1, len(man_opt_gs_loss_rgd))]
                                    man_opt_gs_loss_rgd += [man_opt_gs_loss_rgd[-1]] * (max(0, len_loss - len(man_opt_gs_loss_rgd)))
                                    man_opt_gs_loss_rgd[0] = data[j]['loss']['Grid_search'][ind_b]

                                    man_opt_gs_loss_la_ += torch.as_tensor(man_opt_gs_loss_la, dtype=gtype)
                                    man_opt_gs_loss_rgd_ += torch.as_tensor(man_opt_gs_loss_rgd, dtype=gtype)
                                    ind_b = int(ind_b)+1
                                    ind_b = str(ind_b)
                                else:
                                    hessian_rank_b = hessian_rank_blocks[:, b_ad:b_ad + b, b_ad:b_ad + b]
                                    val = ((hessian_rank_b.squeeze() ** 2).mean(dim=0)).sqrt()
                                    man_opt_gs_loss_la_ += torch.ones(len_loss) * val.squeeze()
                                    man_opt_gs_loss_rgd_ += torch.ones(len_loss) * val.squeeze()
                                b_ad += b
                            c_losses_man_opt_gs_la.append(man_opt_gs_loss_la_)
                            c_losses_man_opt_gs_rgd.append(man_opt_gs_loss_rgd_)
                            losses.append(loss)

                    else:
                        x_0 += 1
                        print('@@@@@@@@@@@@@@@@@@@@@@@@', j, x_0)
                if ind_clamp == len(etas) - 1:
                    if h_size == 1 / 2:
                        c_losses_man_opt_la = torch.stack(c_losses_man_opt_la, dim=0)
                        c_losses_man_opt_rgd = torch.stack(c_losses_man_opt_rgd, dim=0)
                        c_losses_man_opt_la = (c_losses_man_opt_la - torch.as_tensor(losses_mo, dtype=gtype).unsqueeze(dim=1))
                        c_losses_man_opt_rgd = (c_losses_man_opt_rgd - torch.as_tensor(losses_mo, dtype=gtype).unsqueeze(dim=1))
                        c_loss_mean_man_opt[0, :] = c_losses_man_opt_la.mean(dim=0)
                        c_loss_mean_man_opt[1, :] = c_losses_man_opt_rgd.mean(dim=0)

                    c_losses_man_opt_gs_la = torch.stack(c_losses_man_opt_gs_la, dim=0)
                    c_losses_man_opt_gs_rgd = torch.stack(c_losses_man_opt_gs_rgd, dim=0)
                    c_losses_man_opt_gs_la = (c_losses_man_opt_gs_la - torch.as_tensor(losses, dtype=gtype).unsqueeze(dim=1))
                    c_losses_man_opt_gs_rgd = (c_losses_man_opt_gs_rgd - torch.as_tensor(losses, dtype=gtype).unsqueeze(dim=1))

                    c_loss_mean_man_opt_gs[0, :] = torch.mean(c_losses_man_opt_gs_la, 0)
                    c_loss_mean_man_opt_gs[1, :] = torch.mean(c_losses_man_opt_gs_rgd, 0)

            re_colors = ['#9f86fd', '#01796f', '#e4181b', '#00cccc', '#f47bfe']
            h_ind = h_sizes.index(h_size)
            x_tikz = torch.tensor([i for i in range(0, 1001, 50)] + [i for i in range(2000, 50001, 1000)]).cpu()
            h = h_sizes[h_ind]
            if h_size == 1 / 2:
                if ind_clamp == len(etas) - 1:
                    axs[1, 0].plot(x_tikz, c_loss_mean_man_opt[0, :].cpu(),
                                   linestyle='dashed', color='#1B2ACC', label='$RI_{La}$')
                    axs[0, 0].plot(x_tikz, c_loss_mean_man_opt[1, :].cpu(),
                                   color='#1B2ACC', label='$RI_{Rgd}$')
                axs[1, 1].plot(np.array(etas), man_opt_clamp[0, :].cpu() / len(list(data.keys())), linestyle='dashed', color='#1B2ACC', label='$RI_{La}$')
                axs[0, 1].plot(np.array(etas), man_opt_clamp[1, :].cpu() / len(list(data.keys())), color='#1B2ACC', label='$RI_{Rgd}$')
                print(man_opt_clamp[1, :].cpu() / len(list(data.keys())))
            if ind_clamp == len(etas) - 1:
                axs[1, 0].plot(x_tikz, c_loss_mean_man_opt_gs[0, :].cpu(), linestyle='dashed', color=re_colors[h_ind], label='$h_{La} = %.3f$' % h)
                axs[0, 0].plot(x_tikz, c_loss_mean_man_opt_gs[1, :].cpu(), color=re_colors[h_ind], label='$h_{Rgd} = %.3f$' % h)
                print(man_opt_gs_clamp[h_size]['rgd'].cpu()/len(list(data.keys())))
            axs[0, 1].plot(np.array(etas), man_opt_gs_clamp[h_size]['rgd'].cpu() / len(list(data.keys())),
                           color=re_colors[h_ind], label='$h_{Rgd} = %.3f$' % h)
            axs[1, 1].plot(np.array(etas), man_opt_gs_clamp[h_size]['la'].cpu() / len(list(data.keys())), linestyle='dashed',
                           color=re_colors[h_ind], label='$h_{La} = %.3f$' % h)

            axs[0, 0].grid(color='k', linestyle='--', linewidth=0.5)
            axs[0, 1].grid(color='k', linestyle='--', linewidth=0.5)
            axs[1, 0].grid(color='k', linestyle='--', linewidth=0.5)
            axs[1, 1].grid(color='k', linestyle='--', linewidth=0.5)
            axs[0, 0].set_xscale('log')
            axs[0, 1].set_xscale('log')
            axs[1, 0].set_xscale('log')
            axs[1, 1].set_xscale('log')
            axs[0, 0].set_ylabel('Mean $\overline{\ell}_{\mathcal{H}}(U^{(r)})-\overline{\ell}_{\mathcal{H}}(R^T)$')
            axs[1, 0].set_ylabel('Mean $\overline{\ell}_{\mathcal{H}}(U^{(r)})-\overline{\ell}_{\mathcal{H}}(R^T)$')
            axs[0, 1].set_ylabel('Ratio $\mathcal{R}$')
            axs[1, 1].set_ylabel('Ratio $\mathcal{R}$')
            axs[1, 0].set_xlabel('Iterations')
            axs[1, 1].set_xlabel('$\eta$')
    Re_label = axs[0, 0].get_legend_handles_labels()
    La_label = axs[1, 0].get_legend_handles_labels()
    lines = []
    labels = []
    for i in range(len(Re_label[0])):
        lines.append(Re_label[0][i])
        lines.append(La_label[0][i])
        labels.append(Re_label[1][i])
        labels.append(La_label[1][i])
    lines_labels = [(lines, labels)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    print(lines, labels)
    fig.legend(lines, labels, ncol=5, bbox_to_anchor=(.91, 1.), borderpad=0.1)
    noisy_suff = '_noise' if noisy_data else ''
    plt.savefig(report_dir + '/Losses_function{}.png'.format(noisy_suff))

if __name__ == '__main__':
    plot_loss_ratio(etas_clean, names_clean)
    plot_loss_ratio(etas_noisy, names_noisy, noisy_data=True)
    plt.show()
