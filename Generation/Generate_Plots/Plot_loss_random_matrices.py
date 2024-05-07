import json
import torch
from matplotlib import pyplot as plt
from os.path import dirname, abspath
from Utils.Evaluation_utils import loss_function12
#if torch.cuda.is_available():
#    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
out_dir = dirname(dirname(abspath(__file__)))+'/Output_algorithms/Random_matrices/Manifold_optimization'
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
len_loss = {2: 70,
            3: 70,
            4: 70,
            5: 70}
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(9.5, 6))
max_loss_man_opt_2 = 0
max_loss_man_opt_2_noise = 0

loss_mean = 0
losses_inner = []
h_sizes = [1, .5, .25, .125, .1]
labels = ['h={}'.format(i) for i in h_sizes]
markers = [('d', 'thin_diamond'), ('o', 'circle'), ('x', 'x'), ('+', 'plus'), ('H', 'hexagon2'), ('h', 'hexagon1'),
           ('*', 'star'), ('P', 'plus (filled)'), ('p', 'pentagon'), ('s', 'square'), ('8', 'octagon'),
           ('4', 'tri_right'), ('3', 'tri_left'), ('2', 'tri_up'), ('1', 'tri_down'), ('>', 'triangle_right'),
           ('<', 'triangle_left'), ('^', 'triangle_up'), ('v', 'triangle_down'), (',', 'pixel'), ('.', 'point'),
           ('D', 'diamond'), ('_', 'hline'), ('X', 'x (filled)'), ('|', 'vline')]
for ind_names, name in enumerate(names):
    with open('{}/{}'.format(out_dir, name), 'r') as convert_file:
        data = json.load(convert_file)
        c_losses_man_opt_la = []
        c_losses_man_opt_rgd = []
        c_losses_man_opt_la_noise = []
        c_losses_man_opt_rgd_noise = []

        c_losses_man_opt_gs_la = []
        c_losses_man_opt_gs_rgd = []
        c_losses_man_opt_gs_la_noise = []
        c_losses_man_opt_gs_rgd_noise = []

        c_loss_mean_man_opt = torch.zeros(2, len_loss[2])
        c_loss_mean_man_opt_noise = torch.zeros(2, len_loss[2])
        c_loss_std_man_opt = torch.zeros(2, len_loss[2])
        c_loss_std_man_opt_noise = torch.zeros(2, len_loss[2])

        c_loss_mean_man_opt_gs = torch.zeros(2, len_loss[2])
        c_loss_mean_man_opt_gs_noise = torch.zeros(2, len_loss[2])
        c_loss_std_man_opt_gs = torch.zeros(2, len_loss[2])
        c_loss_std_man_opt_gs_noise = torch.zeros(2, len_loss[2])
        print('{}/{}'.format(out_dir, name))
        losses = []
        losses_noise = []
        losses_mo = []
        losses_mo_noise = []
        for j in data.keys():
            d = data[j]['d']
            R = torch.as_tensor(data[j]['R'])
            hessian = torch.as_tensor(data[j]['hessian_basis']['clean'])
            loss = loss_function12(R, hessian, eps=1e-8)
            if 'h_size' not in data[j].keys():
                if len(data[j]['loss']['clean']['Man_Opt_RI']['la']) > 1 and len(data[j]['loss']['clean']['Man_Opt_RI']['rgd']) > 1:
                    man_opt_loss_la = data[j]['loss']['clean']['Man_Opt_RI']['la']
                    man_opt_loss_la = man_opt_loss_la[:min(len_loss[d]+1, len(man_opt_loss_la))]
                    man_opt_loss_la += [man_opt_loss_la[-1]]*(len_loss[d]-len(man_opt_loss_la))
                    c_losses_man_opt_la.append(torch.as_tensor(man_opt_loss_la))

                    man_opt_loss_rgd = data[j]['loss']['clean']['Man_Opt_RI']['rgd']
                    man_opt_loss_rgd = man_opt_loss_rgd[:min(len_loss[d]+1, len(man_opt_loss_rgd))]
                    man_opt_loss_rgd += [man_opt_loss_rgd[-1]]*(
                                    len_loss[d]-len(man_opt_loss_rgd))
                    c_losses_man_opt_rgd.append(torch.as_tensor(man_opt_loss_rgd))
                    losses_mo.append(loss)
                if len(data[j]['loss']['noisy']['Man_Opt_RI']['la']) > 1 and len(data[j]['loss']['noisy']['Man_Opt_RI']['rgd']) > 1:
                    man_opt_loss_la_noise = data[j]['loss']['noisy']['Man_Opt_RI']['la']
                    man_opt_loss_la_noise = man_opt_loss_la_noise[:min(len_loss[d]+1, len(man_opt_loss_la_noise))]
                    man_opt_loss_la_noise += [man_opt_loss_la_noise[-1]] * (max(0, len_loss[d] - len(
                                          man_opt_loss_la_noise)))
                    c_losses_man_opt_la_noise.append(torch.as_tensor(man_opt_loss_la_noise))

                    man_opt_loss_rgd_noise = data[j]['loss']['noisy']['Man_Opt_RI']['rgd']
                    man_opt_loss_rgd_noise = man_opt_loss_rgd_noise[:min(len_loss[d]+1, len(man_opt_loss_rgd_noise))]
                    man_opt_loss_rgd_noise += [man_opt_loss_rgd_noise[-1]] * (max(0, len_loss[d] - len(man_opt_loss_rgd_noise)))

                    c_losses_man_opt_rgd_noise.append(torch.as_tensor(man_opt_loss_rgd_noise))

                    losses_mo_noise.append(loss)
            else:
                if len(data[j]['loss']['clean']['Man_Opt_GS']['la']) > 1 and len(data[j]['loss']['clean']['Man_Opt_GS']['rgd']) > 1:
                    man_opt_gs_loss_la = data[j]['loss']['clean']['Man_Opt_GS']['la']
                    man_opt_gs_loss_la = man_opt_gs_loss_la[:min(len_loss[d]+1, len(man_opt_gs_loss_la))]
                    man_opt_gs_loss_la += [man_opt_gs_loss_la[-1]] * (max(0, len_loss[d]-len(man_opt_gs_loss_la)))
                    man_opt_gs_loss_la[0] = data[j]['loss']['clean']['Grid_search']
                    c_losses_man_opt_gs_la.append(torch.as_tensor(man_opt_gs_loss_la))
                    man_opt_gs_loss_rgd = data[j]['loss']['clean']['Man_Opt_GS']['rgd']
                    man_opt_gs_loss_rgd = man_opt_gs_loss_rgd[:min(len_loss[d]+1, len(man_opt_gs_loss_rgd))]
                    man_opt_gs_loss_rgd += [man_opt_gs_loss_rgd[-1]] * (max(0, len_loss[d]-len(man_opt_gs_loss_rgd)))
                    man_opt_gs_loss_rgd[0] = data[j]['loss']['clean']['Grid_search']
                    c_losses_man_opt_gs_rgd.append(torch.as_tensor(man_opt_gs_loss_rgd))
                    losses.append(loss)
                if len(data[j]['loss']['noisy']['Man_Opt_GS']['la']) > 1 and len(data[j]['loss']['noisy']['Man_Opt_GS']['rgd']) > 1:
                    man_opt_gs_loss_la_noise = data[j]['loss']['noisy']['Man_Opt_GS']['la']
                    man_opt_gs_loss_la_noise = man_opt_gs_loss_la_noise[:min(len_loss[d]+1, len(man_opt_gs_loss_la_noise))]
                    man_opt_gs_loss_la_noise += [man_opt_gs_loss_la_noise[-1]] * (max(0, len_loss[d] - len(man_opt_gs_loss_la_noise)))
                    man_opt_gs_loss_la_noise[0] = data[j]['loss']['noisy']['Grid_search']
                    c_losses_man_opt_gs_la_noise.append(torch.as_tensor(man_opt_gs_loss_la_noise))
                    man_opt_gs_loss_rgd_noise = data[j]['loss']['noisy']['Man_Opt_GS']['rgd']
                    man_opt_gs_loss_rgd_noise = man_opt_gs_loss_rgd_noise[:min(len_loss[d]+1, len(man_opt_gs_loss_rgd_noise))]
                    man_opt_gs_loss_rgd_noise += [man_opt_gs_loss_rgd_noise[-1]] * (max(0, len_loss[d] - len(man_opt_gs_loss_rgd_noise)))
                    man_opt_gs_loss_rgd_noise[0] = data[j]['loss']['noisy']['Grid_search']
                    c_losses_man_opt_gs_rgd_noise.append(torch.as_tensor(man_opt_gs_loss_rgd_noise))
                    losses_noise.append(loss)
        if 'h_size' not in data[j].keys():
            c_losses_man_opt_la = torch.stack(c_losses_man_opt_la, dim=0)
            c_losses_man_opt_rgd = torch.stack(c_losses_man_opt_rgd, dim=0)
            c_losses_man_opt_la_noise = torch.stack(c_losses_man_opt_la_noise, dim=0)
            c_losses_man_opt_rgd_noise = torch.stack(c_losses_man_opt_rgd_noise, dim=0)

            print('\n c_losses_la_noise: ', c_losses_man_opt_la_noise.shape, c_losses_man_opt_rgd_noise.shape)
            c_losses_man_opt_la = (c_losses_man_opt_la-torch.as_tensor(losses_mo).unsqueeze(dim=1)).abs()
            c_losses_man_opt_rgd = (c_losses_man_opt_rgd-torch.as_tensor(losses_mo).unsqueeze(dim=1)).abs()

            c_loss_mean_man_opt[0, :] = c_losses_man_opt_la.mean(dim=0)
            c_loss_mean_man_opt[1, :] = c_losses_man_opt_rgd.mean(dim=0)

            c_losses_man_opt_la_noise = (c_losses_man_opt_la_noise - torch.as_tensor(losses_mo_noise).unsqueeze(dim=1)).abs()
            c_losses_man_opt_rgd_noise = (c_losses_man_opt_rgd_noise - torch.as_tensor(losses_mo_noise).unsqueeze(dim=1)).abs()
            print('\n losses: ', c_loss_mean_man_opt_noise[0, :].shape, torch.median(c_losses_man_opt_la_noise, dim=0)[0].shape)
            c_loss_mean_man_opt_noise[0, :] = torch.mean(c_losses_man_opt_la_noise, dim=0)
            c_loss_mean_man_opt_noise[1, :] = c_losses_man_opt_rgd_noise.mean(dim=0)
        else:
            c_losses_man_opt_gs_la = torch.stack(c_losses_man_opt_gs_la, dim=0)
            c_losses_man_opt_gs_rgd = torch.stack(c_losses_man_opt_gs_rgd, dim=0)
            c_losses_man_opt_gs_la_noise = torch.stack(c_losses_man_opt_gs_la_noise, dim=0)
            c_losses_man_opt_gs_rgd_noise = torch.stack(c_losses_man_opt_gs_rgd_noise, dim=0)

            c_losses_man_opt_gs_la = (c_losses_man_opt_gs_la - torch.as_tensor(losses).unsqueeze(dim=1)).abs()
            c_losses_man_opt_gs_rgd = (c_losses_man_opt_gs_rgd - torch.as_tensor(losses).unsqueeze(dim=1)).abs()
            c_loss_mean_man_opt_gs[0, :] = torch.mean(c_losses_man_opt_gs_la, 0)
            c_loss_mean_man_opt_gs[1, :] = torch.mean(c_losses_man_opt_gs_rgd, 0)

            c_losses_man_opt_gs_la_noise = (c_losses_man_opt_gs_la_noise - torch.as_tensor(losses_noise).unsqueeze(dim=1)).abs()
            c_losses_man_opt_gs_rgd_noise = (c_losses_man_opt_gs_rgd_noise - torch.as_tensor(losses_noise).unsqueeze(dim=1)).abs()
            c_loss_mean_man_opt_gs_noise[0, :] = c_losses_man_opt_gs_la_noise.mean(dim=0)
            c_loss_mean_man_opt_gs_noise[1, :] = c_losses_man_opt_gs_rgd_noise.mean(dim=0)

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

        re_colors = ['#9f86fd', '#01796f', '#e4181b' , '#00cccc', '#f47bfe']
        re_colors_fills = ['#FAE6FA', '#f0fff0', '#ff91a4', '#00cccc', '#f47bfe']
        x_tikz = torch.tensor([i for i in range(0, 1001, 50)] + [i for i in range(2000, 50001, 1000)]).cpu()
        print(x_tikz.shape)
        if 'h_size' not in data[str(0)].keys():
            ax.plot(x_tikz, c_loss_mean_man_opt[0, :].cpu(), linestyle='dashed', color='#1B2ACC', label='$RI_{La}$')
            ax.plot(x_tikz, c_loss_mean_man_opt[1, :].cpu(), color='#1B2ACC', label='$RI_{Rgd}$')
            ax2.plot(x_tikz, c_loss_mean_man_opt_noise[0, :].cpu(), linestyle='dashed', color='#1B2ACC', label='$RI_{La}$')
            ax2.plot(x_tikz, c_loss_mean_man_opt_noise[1, :].cpu(), color='#1B2ACC', label='$RI_{Rgd}$')
        else:
            h_ind = h_sizes.index(data[str(0)]['h_size'])
            m, l = markers[h_ind]
            h = h_sizes[h_ind]
            ax.plot(x_tikz, c_loss_mean_man_opt_gs[0, :].cpu(), linestyle='dashed',
                    color=re_colors[h_ind], label='$h_{La} = %.3f$'%h)
            ax.plot(x_tikz, c_loss_mean_man_opt_gs[1, :].cpu(), color=re_colors[h_ind], label='$h_{Rgd} = %.3f$' % h)

            ax2.plot(x_tikz, c_loss_mean_man_opt_gs_noise[0, :].cpu(), linestyle='dashed', color=re_colors[h_ind],
                     label='$h_{La} = %.3f$'%h)
            ax2.plot(x_tikz, c_loss_mean_man_opt_gs_noise[1, :].cpu(), color=re_colors[h_ind],
                     label='$h_{Rgd} = %.3f$'%h)
            ax.set_title('$d={}$'.format(d))
        ax.grid(color='k', linestyle='--', linewidth=0.5)
        ax2.grid(color='k', linestyle='--', linewidth=0.5)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xscale('log')

        axs[0, 0].set_ylabel('Mean $\ell_{\mathcal{H}}(U^{(r)})-\ell_{\mathcal{H}}(R^T)$')
        axs[1, 0].set_ylabel('Mean $\ell_{\mathcal{H}}(U^{(r)})-\ell_{\mathcal{H}}(R^T)$')

lines_labels = [axs[0, 0].get_legend_handles_labels()]

lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, ncol=6, bbox_to_anchor=(.91, 1.), borderpad=0.1)
for n_r in range(2):
    for n_c in range(4):
        box = axs[n_r, n_c].get_position()
        box.y0 = box.y0 - 0.02
        box.y1 = box.y1 - 0.02
        if n_c >= 1:
            box.x0 = box.x0 + n_c*0.02
            box.x1 = box.x1 + n_c*0.02
        axs[n_r, n_c].set_position(box)
        if n_r == 1:
            axs[n_r, n_c].set_xlabel('Iterations')

root = dirname(dirname(abspath(__file__))) + '/Plots/Random_matrices'
plt.savefig(root + '/Losses.png')
plt.show()

