import copy
import json
import torch
import numpy as np
from matplotlib import pyplot as plt
from os.path import dirname, abspath
from Utils.Evaluation_utils import compute_hessian_rotmatrix
from Libs.Grid_Search import Method

etas = [1e-8, 1e-7, 1e-8, 1e-9,  1e-10, 1e-11]
etas_noise = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
clamp_gt = 1e-6
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
out_dir = dirname(dirname(abspath(__file__)))+'/Output_algorithms/Random_matrices/Manifold_optimization/Corrections'
report_dir = dirname(dirname(abspath(__file__))) + '/Plots/Random_matrices'
names = [
        'Compare_Man_opt_grid_search_100_dim_2.json',
        'Compare_Man_opt_grid_search_100_h_1.0_bh_3.14_dim_2.json',
        'Compare_Man_opt_grid_search_100_h_0.5_bh_3.14_dim_2.json',
        'Compare_Man_opt_grid_search_100_h_0.25_bh_3.14_dim_2.json',
        'Compare_Man_opt_grid_search_100_h_0.125_bh_3.14_dim_2.json',
        'Compare_Man_opt_grid_search_100_h_0.1_bh_3.14_dim_2.json',
        'Compare_Man_opt_grid_search_100_dim_3.json',
        'Compare_Man_opt_grid_search_100_h_1.0_bh_1.57_dim_3.json',
        'Compare_Man_opt_grid_search_100_h_0.5_bh_1.57_dim_3.json',
        'Compare_Man_opt_grid_search_100_h_0.25_bh_1.57_dim_3.json',
        'Compare_Man_opt_grid_search_100_h_0.125_bh_1.57_dim_3.json',
        'Compare_Man_opt_grid_search_100_h_0.1_bh_1.57_dim_3.json',
        'Compare_Man_opt_grid_search_100_dim_4.json',
        'Compare_Man_opt_grid_search_100_h_1.0_bh_1.57_dim_4.json',
        'Compare_Man_opt_grid_search_100_h_0.5_bh_1.57_dim_4.json',
        'Compare_Man_opt_grid_search_100_h_0.25_bh_1.57_dim_4.json',
        'Compare_Man_opt_grid_search_100_h_0.125_bh_1.57_dim_4.json',
        'Compare_Man_opt_grid_search_100_h_0.1_bh_1.00_dim_4.json',
        'Compare_Man_opt_grid_search_100_dim_5.json',
        'Compare_Man_opt_grid_search_100_h_1.0_bh_2.36_dim_5.json'
            ]


markers = [('d', 'thin_diamond'), ('o', 'circle'), ('x', 'x'), ('+', 'plus'), ('H', 'hexagon2'), ('h', 'hexagon1'),
            ('*', 'star'), ('P', 'plus (filled)'), ('p', 'pentagon'), ('s', 'square'), ('8', 'octagon'),
            ('4', 'tri_right'), ('3', 'tri_left'), ('2', 'tri_up'), ('1', 'tri_down'), ('>', 'triangle_right'),
            ('<', 'triangle_left'), ('^', 'triangle_up'), ('v', 'triangle_down'), (',', 'pixel'), ('.', 'point'),
            ('D', 'diamond'), ('_', 'hline'), ('X', 'x (filled)'), ('|', 'vline')]

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))

man_opt_clamp = torch.zeros(8, len(etas))
man_opt_clamp_noise = torch.zeros(8, len(etas_noise))


man_opt_gs_clamp = {}
h_sizes = [1, .5, .25, .125, .1]
dims = [2, 3, 4, 5]
hs_dim = {2: [1, 1/2, 1/4, 1/8, 1/10],
          3: [1, 1/2, 1/4, 1/8, 1/10],
          4: [1, 1/2, 1/4, 1/8, 1/10],
          5: [1],
          }
man_opt = {}
man_opt_noise = {}
cls = [5e-5, 1e-5, 1e-6]
cls_noise = [5e-4, 2e-4, 1e-4]

for i in range(len(cls)):
    man_opt[cls[i]] = {}
    man_opt_noise[cls_noise[i]] = {}
    for d in hs_dim.keys():
        man_opt[cls[i]][d] = {'la': {}, 'rgd': {}}
        man_opt_noise[cls_noise[i]][d] = {'la': {}, 'rgd': {}}

        man_opt[cls[i]][d]['la'][0] = []
        man_opt[cls[i]][d]['rgd'][0] = []

        man_opt_noise[cls_noise[i]][d]['la'][0] = []
        man_opt_noise[cls_noise[i]][d]['rgd'][0] = []

        for h in hs_dim[d]:
            man_opt[cls[i]][d]['la'][h] = []
            man_opt[cls[i]][d]['rgd'][h] = []

            man_opt_noise[cls_noise[i]][d]['la'][h] = []
            man_opt_noise[cls_noise[i]][d]['rgd'][h] = []

for d in hs_dim.keys():
    man_opt_gs_clamp[d] = {}
    for h in hs_dim[d]:
        man_opt_gs_clamp[d][h] = {}
        man_opt_gs_clamp[d][h]['clean'] = {}
        man_opt_gs_clamp[d][h]['noisy'] = {}

        man_opt_gs_clamp[d][h]['clean']['la'] = torch.zeros(len(etas))
        man_opt_gs_clamp[d][h]['clean']['rgd'] = torch.zeros(len(etas))

        man_opt_gs_clamp[d][h]['noisy']['la'] = torch.zeros(len(etas_noise))
        man_opt_gs_clamp[d][h]['noisy']['rgd'] = torch.zeros(len(etas_noise))
for name in names:
    #d=0
    with open('{}/{}'.format(
            out_dir, name), 'r') as convert_file:
        data = json.load(convert_file)
        print('\n dim {} name: {} '.format(data[list(data.keys())[0]]['d'], name))
        for ind_clamp, clamp in enumerate(etas):
            for j in data.keys():# range(43):#
                d = data[j]['d']
                ind_dim = dims.index(d)
                R = torch.as_tensor(data[j]['R'])
                H_gt = ((R @ torch.as_tensor(data[j]['hessian']['clean']) @ R.T).abs()).mean(dim=0)
                H_gt[H_gt != torch.clamp(H_gt, clamp_gt)] = 0
                zero_norm_gt = len(H_gt.nonzero())
                method = Method.Manifold_Opt if 'h_size' not in data[j].keys() else Method.Manifold_Opt_GS
                if 'h_size' not in data[j].keys():
                    H_man_opt_re, H_man_opt_la = compute_hessian_rotmatrix(data[j], method)
                    H_man_opt_la[H_man_opt_la != torch.clamp(H_man_opt_la, clamp)] = 0
                    zero_norm_H_man_opt_la = len(H_man_opt_la.nonzero())
                    if zero_norm_H_man_opt_la - zero_norm_gt > 0:
                        man_opt_clamp[ind_dim, ind_clamp] += 1
                    H_man_opt_re[H_man_opt_re != torch.clamp(H_man_opt_re, clamp)] = 0
                    zero_norm_H_man_opt_re = len(H_man_opt_re.nonzero())
                    if zero_norm_H_man_opt_re - zero_norm_gt > 0:
                        man_opt_clamp[ind_dim+len(dims), ind_clamp] += 1

                else:
                    h_size = data[j]['h_size']
                    H_man_opt_gs_re, H_man_opt_gs_la = compute_hessian_rotmatrix(data[j], method)
                    H_man_opt_gs_la[H_man_opt_gs_la != torch.clamp(H_man_opt_gs_la, clamp)] = 0
                    zero_norm_H_man_opt_gs_la = len(H_man_opt_gs_la.nonzero())
                    if zero_norm_H_man_opt_gs_la - zero_norm_gt > 0:
                        man_opt_gs_clamp[d][h_size]['clean']['la'][ind_clamp] += 1

                    H_man_opt_gs_re[H_man_opt_gs_re != torch.clamp(H_man_opt_gs_re, clamp)] = 0
                    zero_norm_H_man_opt_gs_re = len(H_man_opt_gs_re.nonzero())
                    if zero_norm_H_man_opt_gs_re - zero_norm_gt > 0:
                        man_opt_gs_clamp[d][h_size]['clean']['rgd'][ind_clamp] += 1


        for ind_clamp, clamp in enumerate(etas_noise):
            for j in data.keys():
                d = data[j]['d']
                ind_dim = dims.index(d)
                R = torch.as_tensor(data[j]['R'])
                H_gt = (R @ torch.as_tensor(data[j]['hessian_basis']['clean']) @ R.T).abs().mean(dim=0)
                H_gt[H_gt != torch.clamp(H_gt, clamp)] = 0
                zero_norm_gt = len(H_gt.nonzero())
                if 'h_size' not in data[j].keys():
                    H_man_opt_re_noise, H_man_opt_la_noise = compute_hessian_rotmatrix(data[j], method,
                                                                                             noisy_rot=True)
                    H_man_opt_la_noise[H_man_opt_la_noise != torch.clamp(H_man_opt_la_noise, clamp)] = 0
                    zero_norm_H_man_opt_la_noise = len(H_man_opt_la_noise.nonzero())
                    if zero_norm_H_man_opt_la_noise - zero_norm_gt > 0:
                        man_opt_clamp_noise[ind_dim, ind_clamp] += 1

                        if clamp in man_opt_noise.keys():
                            man_opt_noise[clamp][d]['la'][0] += [zero_norm_H_man_opt_la_noise - zero_norm_gt]

                    H_man_opt_re_noise[H_man_opt_re_noise != torch.clamp(H_man_opt_re_noise, clamp)] = 0
                    zero_norm_H_man_opt_re_noise = len(H_man_opt_re_noise.nonzero())
                    if zero_norm_H_man_opt_re_noise - zero_norm_gt > 0:
                        man_opt_clamp_noise[ind_dim+len(dims), ind_clamp] += 1

                        if clamp in man_opt_noise.keys():
                            man_opt_noise[clamp][d]['rgd'][0] += [zero_norm_H_man_opt_re_noise - zero_norm_gt]
                else:
                    h_size = data[j]['h_size']
                    H_man_opt_gs_re_noise, H_man_opt_gs_la_noise = compute_hessian_rotmatrix(data[j], method,
                                                                                             noisy_rot=True)
                    H_man_opt_gs_la_noise[H_man_opt_gs_la_noise != torch.clamp(H_man_opt_gs_la_noise, clamp)] = 0
                    zero_norm_H_man_opt_gs_la_noise = len(H_man_opt_gs_la_noise.nonzero())
                    if zero_norm_H_man_opt_gs_la_noise - zero_norm_gt > 0:
                        man_opt_gs_clamp[d][h_size]['noisy']['la'][ind_clamp] +=1
                        if clamp in man_opt_noise.keys():
                            man_opt_noise[clamp][d]['la'][h_size] += [zero_norm_H_man_opt_gs_la_noise - zero_norm_gt]

                    H_man_opt_gs_re_noise[H_man_opt_gs_re_noise != torch.clamp(H_man_opt_gs_re_noise, clamp)] = 0
                    zero_norm_H_man_opt_gs_re_noise = len(H_man_opt_gs_re_noise.nonzero())
                    if zero_norm_H_man_opt_gs_re_noise - zero_norm_gt > 0:
                        man_opt_gs_clamp[d][h_size]['noisy']['rgd'][ind_clamp] += 1
                        if clamp in man_opt_noise.keys():
                            man_opt_noise[clamp][d]['rgd'][h_size] += [zero_norm_H_man_opt_gs_re_noise - zero_norm_gt]

        if d == 2:
            ax = axs[0, 0]
            ax2 = axs[1, 0]
        elif d == 3:
            ax = axs[0, 1]
            ax2 = axs[1, 1]
        elif d == 4:
            ax = axs[0, 2]
            ax2 = axs[1, 2]
        else:
            ax = axs[0, 3]
            ax2 = axs[1, 3]
        re_colors = ['#9f86fd', '#01796f', '#e4181b', '#00cccc', '#f47bfe']
        re_colors_fills = ['#FAE6FA', '#f0fff0', '#ff91a4', '#00cccc', '#f47bfe']
        if 'h_size' not in data[j].keys():
            ax.plot(np.array(etas), man_opt_clamp[dims.index(d), :].cpu()/len(list(data.keys())),
                    linestyle='dashed', color='#1B2ACC', label='$RI_{La}$')
            ax.plot(np.array(etas), man_opt_clamp[dims.index(d)+len(dims), :].cpu()/len(list(data.keys())),
                    color='#1B2ACC', label='$RI_{Rgd}$')

            ax2.plot(np.array(etas_noise), man_opt_clamp_noise[dims.index(d)].cpu()/len(list(data.keys())),
                     linestyle='dashed', color='#1B2ACC', label='$RI_{La}$')
            ax2.plot(np.array(etas_noise), man_opt_clamp_noise[dims.index(d)+len(dims)].cpu()/len(list(data.keys())),
                     color='#1B2ACC', label='$RI_{Rgd}$')

        else:
            h_ind = h_sizes.index(data[str(0)]['h_size'])
            m, l = markers[h_ind]
            h = h_sizes[h_ind]
            ax.plot(np.array(etas), man_opt_gs_clamp[d][h_size]['clean']['la'].cpu() / len(list(data.keys())), linestyle='dashed',
                        color=re_colors[h_ind], label='$h_{La} = %.3f$'%h)
            ax.plot(np.array(etas), man_opt_gs_clamp[d][h_size]['clean']['rgd'].cpu() / len(list(data.keys())), color=re_colors[h_ind],
                    label='$h_{Rgd} = %.3f$'%h)#, label='hre %.2f$'%h_size

            ax2.plot(np.array(etas_noise), man_opt_gs_clamp[d][h_size]['noisy']['la'].cpu() / len(list(data.keys())),
                     linestyle='dashed', color=re_colors[h_ind], label='$h_{La} = %.3f$'%h)
            ax2.plot(np.array(etas_noise), man_opt_gs_clamp[d][h_size]['noisy']['rgd'].cpu() / len(list(data.keys())),
                     color=re_colors[h_ind], label='$h_{Rgd} = %.3f$'%h)
        axs[0,dims.index(d)].set_title('$d={}$'.format(d))


        ax.grid(color='k', linestyle='--', linewidth=0.5)
        ax2.grid(color='k', linestyle='--', linewidth=0.5)
        ax2.set_xscale('log')
    ax.set_xscale('log')
    axs[0, 0].set_ylabel('Ratio $\mathcal{R}$')
    axs[1, 0].set_ylabel('Ratio $\mathcal{R}$')

for n_r in range(2):
    for n_c in range(4):
        #print(axs[n_r, n_c].get_position())
        box = axs[n_r, n_c].get_position()
        box.y0 = box.y0 - 0.02
        box.y1 = box.y1 - 0.02
        if n_c >= 1:
            box.x0 = box.x0 + n_c * 0.02
            box.x1 = box.x1 + n_c * 0.02
        axs[n_r, n_c].set_position(box)
        if n_r == 1:
            axs[n_r, n_c].set_xlabel('$\eta$')

box = axs[1, 3].get_position()
box.x0 = box.x0 + 0.01
box.x1 = box.x1 + 0.01
axs[1, 3].set_position(box)

lines_labels = [axs[0, 0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, ncol=6, bbox_to_anchor=(.9, 1.), borderpad=0.1)

plt.savefig(report_dir + '/Truncation.png')
plt.show()