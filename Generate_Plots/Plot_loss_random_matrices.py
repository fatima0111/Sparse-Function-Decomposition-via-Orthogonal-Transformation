import json
import torch
from matplotlib import pyplot as plt
from os.path import dirname, abspath

out_dir = "/store/steinitz/datastore/fatimaba/Github/trafo_nova/Anova_AE/Out_comp/all"
report_dir = "/store/steinitz/datastore/fatimaba/Github/trafo_nova/Anova_AE/Test_Cases_Plots"
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
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(7, 6))
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
        c_losses_man_opt_re = []
        c_losses_man_opt_la_noise = []
        c_losses_man_opt_re_noise = []

        c_losses_man_opt_gs_la = []
        c_losses_man_opt_gs_re = []
        c_losses_man_opt_gs_la_noise = []
        c_losses_man_opt_gs_re_noise = []

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
            dim = data[j]['dim']
            v = torch.as_tensor(data[j]['v'])
            hessian = torch.as_tensor(data[j]['svd_basis'])
            res = torch.matmul(torch.matmul(v, hessian), v.t())
            res2 = ((res.abs() ** 2).mean(dim=0) + 1e-8).sqrt()
            loss = torch.norm(res2, p=1)
            if 'h_size' not in data[j].keys():
                if len(data[j]['loss']['gt']['Man_Opt']['la']) > 1 and len(data[j]['loss']['gt']['Man_Opt']['re']) > 1:
                    man_opt_loss_la = data[j]['loss']['gt']['Man_Opt']['la']
                    man_opt_loss_la = man_opt_loss_la[:min(len_loss[dim]+1, len(man_opt_loss_la))]
                    man_opt_loss_la += [man_opt_loss_la[-1]]*(len_loss[dim]-len(man_opt_loss_la))
                    c_losses_man_opt_la.append(torch.as_tensor(man_opt_loss_la))

                    man_opt_loss_re = data[j]['loss']['gt']['Man_Opt']['re']
                    man_opt_loss_re = man_opt_loss_re[:min(len_loss[dim]+1, len(man_opt_loss_re))]
                    man_opt_loss_re += [man_opt_loss_re[-1]]*(
                                    len_loss[dim]-len(man_opt_loss_re))
                    c_losses_man_opt_re.append(torch.as_tensor(man_opt_loss_re))
                    losses_mo.append(loss)
                if len(data[j]['loss']['noise']['Man_Opt']['la']) > 1 and len(data[j]['loss']['noise']['Man_Opt']['re']) > 1:
                    man_opt_loss_la_noise = data[j]['loss']['noise']['Man_Opt']['la']
                    man_opt_loss_la_noise = man_opt_loss_la_noise[:min(len_loss[dim]+1, len(man_opt_loss_la_noise))]
                    man_opt_loss_la_noise += [man_opt_loss_la_noise[-1]] * (max(0, len_loss[dim] - len(
                                          man_opt_loss_la_noise)))
                    c_losses_man_opt_la_noise.append(torch.as_tensor(man_opt_loss_la_noise))

                    man_opt_loss_re_noise = data[j]['loss']['noise']['Man_Opt']['re']
                    man_opt_loss_re_noise = man_opt_loss_re_noise[:min(len_loss[dim]+1, len(man_opt_loss_re_noise))]
                    man_opt_loss_re_noise += [man_opt_loss_re_noise[-1]] * (max(0, len_loss[dim] - len(man_opt_loss_re_noise)))

                    c_losses_man_opt_re_noise.append(torch.as_tensor(man_opt_loss_re_noise))

                    losses_mo_noise.append(loss)
            else:
                if len(data[j]['loss']['gt']['Man_Opt_GS']['la']) > 1 and len(data[j]['loss']['gt']['Man_Opt_GS']['re']) > 1:
                    man_opt_gs_loss_la = data[j]['loss']['gt']['Man_Opt_GS']['la']
                    man_opt_gs_loss_la = man_opt_gs_loss_la[:min(len_loss[dim]+1, len(man_opt_gs_loss_la))]
                    man_opt_gs_loss_la += [man_opt_gs_loss_la[-1]] * (max(0, len_loss[dim]-len(man_opt_gs_loss_la)))
                    man_opt_gs_loss_la[0] = data[j]['loss']['gt']['Grid_search']
                    c_losses_man_opt_gs_la.append(torch.as_tensor(man_opt_gs_loss_la))
                    man_opt_gs_loss_re = data[j]['loss']['gt']['Man_Opt_GS']['re']
                    man_opt_gs_loss_re = man_opt_gs_loss_re[:min(len_loss[dim]+1, len(man_opt_gs_loss_re))]
                    man_opt_gs_loss_re += [man_opt_gs_loss_re[-1]] * (max(0, len_loss[dim]-len(man_opt_gs_loss_re)))
                    man_opt_gs_loss_re[0] = data[j]['loss']['gt']['Grid_search']
                    c_losses_man_opt_gs_re.append(torch.as_tensor(man_opt_gs_loss_re))
                    losses.append(loss)
                if len(data[j]['loss']['noise']['Man_Opt_GS']['la']) > 1 and len(data[j]['loss']['noise']['Man_Opt_GS']['re']) > 1:
                    man_opt_gs_loss_la_noise = data[j]['loss']['noise']['Man_Opt_GS']['la']
                    man_opt_gs_loss_la_noise = man_opt_gs_loss_la_noise[:min(len_loss[dim]+1, len(man_opt_gs_loss_la_noise))]
                    man_opt_gs_loss_la_noise += [man_opt_gs_loss_la_noise[-1]] * (max(0, len_loss[dim] - len(man_opt_gs_loss_la_noise)))
                    man_opt_gs_loss_la_noise[0] = data[j]['loss']['noise']['Grid_search']
                    c_losses_man_opt_gs_la_noise.append(torch.as_tensor(man_opt_gs_loss_la_noise))
                    man_opt_gs_loss_re_noise = data[j]['loss']['noise']['Man_Opt_GS']['re']
                    man_opt_gs_loss_re_noise = man_opt_gs_loss_re_noise[:min(len_loss[dim]+1, len(man_opt_gs_loss_re_noise))]
                    man_opt_gs_loss_re_noise +=[man_opt_gs_loss_re_noise[-1]] * (max(0, len_loss[dim] - len(man_opt_gs_loss_re_noise)))
                    man_opt_gs_loss_re_noise[0] = data[j]['loss']['noise']['Grid_search']
                    c_losses_man_opt_gs_re_noise.append(torch.as_tensor(man_opt_gs_loss_re_noise))
                    losses_noise.append(loss)
        if 'h_size' not in data[j].keys():
            c_losses_man_opt_la = torch.stack(c_losses_man_opt_la, dim=0)
            c_losses_man_opt_re = torch.stack(c_losses_man_opt_re, dim=0)
            c_losses_man_opt_la_noise = torch.stack(c_losses_man_opt_la_noise, dim=0)
            c_losses_man_opt_re_noise = torch.stack(c_losses_man_opt_re_noise, dim=0)

            print('\n c_losses_la_noise: ', c_losses_man_opt_la_noise.shape, c_losses_man_opt_re_noise.shape)
            c_losses_man_opt_la = (c_losses_man_opt_la-torch.as_tensor(losses_mo).unsqueeze(dim=1)).abs()
            c_losses_man_opt_re = (c_losses_man_opt_re-torch.as_tensor(losses_mo).unsqueeze(dim=1)).abs()

            c_loss_mean_man_opt[0, :] = c_losses_man_opt_la.mean(dim=0)
            c_loss_mean_man_opt[1, :] = c_losses_man_opt_re.mean(dim=0)

            c_losses_man_opt_la_noise = (c_losses_man_opt_la_noise - torch.as_tensor(losses_mo_noise).unsqueeze(dim=1)).abs()
            c_losses_man_opt_re_noise = (c_losses_man_opt_re_noise - torch.as_tensor(losses_mo_noise).unsqueeze(dim=1)).abs()
            print('\n losses: ', c_loss_mean_man_opt_noise[0, :].shape, torch.median(c_losses_man_opt_la_noise, dim=0)[0].shape)
            c_loss_mean_man_opt_noise[0, :] = torch.mean(c_losses_man_opt_la_noise, dim=0)
            c_loss_mean_man_opt_noise[1, :] = c_losses_man_opt_re_noise.mean(dim=0)
        else:
            c_losses_man_opt_gs_la = torch.stack(c_losses_man_opt_gs_la, dim=0)
            c_losses_man_opt_gs_re = torch.stack(c_losses_man_opt_gs_re, dim=0)
            c_losses_man_opt_gs_la_noise = torch.stack(c_losses_man_opt_gs_la_noise, dim=0)
            c_losses_man_opt_gs_re_noise = torch.stack(c_losses_man_opt_gs_re_noise, dim=0)

            c_losses_man_opt_gs_la = (c_losses_man_opt_gs_la - torch.as_tensor(losses).unsqueeze(dim=1)).abs()
            c_losses_man_opt_gs_re = (c_losses_man_opt_gs_re - torch.as_tensor(losses).unsqueeze(dim=1)).abs()
            c_loss_mean_man_opt_gs[0, :] = torch.mean(c_losses_man_opt_gs_la, 0)
            c_loss_mean_man_opt_gs[1, :] = torch.mean(c_losses_man_opt_gs_re, 0)

            c_losses_man_opt_gs_la_noise = (c_losses_man_opt_gs_la_noise - torch.as_tensor(losses_noise).unsqueeze(dim=1)).abs()
            c_losses_man_opt_gs_re_noise = (c_losses_man_opt_gs_re_noise - torch.as_tensor(losses_noise).unsqueeze(dim=1)).abs()
            c_loss_mean_man_opt_gs_noise[0, :] = c_losses_man_opt_gs_la_noise.mean(dim=0)
            c_loss_mean_man_opt_gs_noise[1, :] = c_losses_man_opt_gs_re_noise.mean(dim=0)

        if dim == 2:
            ax = axs[0, 0]
            ax2 = axs[1, 0]

        elif dim == 3:
            ax = axs[0, 1]
            ax2 = axs[1, 1]
        elif dim == 4:
            ax = axs[0, 2]
            ax2 = axs[1, 2]
        else:
            ax = axs[0, 3]
            ax2 = axs[1, 3]

        re_colors = ['#9f86fd', '#01796f', '#e4181b' , '#00cccc', '#f47bfe']
        re_colors_fills = ['#FAE6FA', '#f0fff0', '#ff91a4', '#00cccc', '#f47bfe']
        x_tiksz = torch.tensor([i for i in range(0, 1001, 50)]+[i for i in range(2000, 50001, 1000)])
        print(x_tiksz.shape)
        if 'h_size' not in data[str(0)].keys():
            ax.plot(x_tiksz, c_loss_mean_man_opt[0, :], linestyle='dashed', color='#1B2ACC', label='$RI_{La}$')
            ax.plot(x_tiksz, c_loss_mean_man_opt[1, :], color='#1B2ACC', label='$RI_{Rgd}$')
            ax2.plot(x_tiksz, c_loss_mean_man_opt_noise[0, :], linestyle='dashed', color='#1B2ACC', label='$RI_{La}$')
            ax2.plot(x_tiksz, c_loss_mean_man_opt_noise[1, :],  color='#1B2ACC', label='$RI_{Rgd}$')
        else:
            h_ind = h_sizes.index(data[str(0)]['h_size'])
            m, l = markers[h_ind]
            h = h_sizes[h_ind]
            ax.plot(x_tiksz, c_loss_mean_man_opt_gs[0, :], linestyle='dashed',
                    color=re_colors[h_ind], label='$h_{La} = %.3f$'%h)
            ax.plot(x_tiksz, c_loss_mean_man_opt_gs[1, :], color=re_colors[h_ind], label='$h_{Rgd} = %.3f$'%h)

            ax2.plot(x_tiksz, c_loss_mean_man_opt_gs_noise[0, :], linestyle='dashed', color=re_colors[h_ind],
                    label='$h_{La} = %.3f$'%h)
            ax2.plot(x_tiksz, c_loss_mean_man_opt_gs_noise[1, :], color=re_colors[h_ind],
                    label='$h_{Rgd} = %.3f$'%h)
            ax.set_title('$d={}$'.format(dim))
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
        print(axs[n_r, n_c].get_position())
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

