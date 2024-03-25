import json
import torch
from Utils.Evaluation_utils import compute_hessian_rotmatrix
from Libs.Grid_Search import Method
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
clamp = 1e-9
clamp_noise = 1e-4
clamp_gt = 1e-6
out_dir = "/store/steinitz/datastore/fatimaba/Github/trafo_nova/Anova_AE/Out_comp"
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

Non_zeros = [[], [], [], []]
report = {}
for name in names:
    Norm_diff_la = []
    Norm_diff_re = []

    Norm_diff_la_noise = []
    Norm_diff_re_noise = []

    Time_la = []
    Time_re = []

    Time_la_noise = []
    Time_re_noise = []

    non_zero_man_opt_la = []
    non_zero_man_opt_gs_la = []

    non_zero_man_opt_re = []
    non_zero_man_opt_gs_re = []

    non_zero_man_opt_la_noise = []
    non_zero_man_opt_gs_la_noise = []

    non_zero_man_opt_re_noise = []
    non_zero_man_opt_gs_re_noise = []

    man_opt_gs_count = {'gt': {'re': {1: 0, 2: 0, 3: 0},
                             'la': {1: 0, 2: 0, 3: 0}},
                        'noise': {'re': {1: 0, 2: 0, 3: 0},
                                'la': {1: 0, 2: 0, 3: 0}}}
    man_opt_count = {'gt': {'re': {1: 0, 2: 0, 3: 0},
                             'la': {1: 0, 2: 0, 3: 0}},
                     'noise': {'re': {1: 0, 2: 0, 3: 0},
                                'la': {1: 0, 2: 0, 3: 0}}}
    non_zero_gs = []
    non_zero_gs_noise = []
    sparse = []
    with open('{}/{}'.format(
            out_dir, name), 'r') as convert_file:
        data = json.load(convert_file)
        print('\n dim {} name: {} '.format(data[list(data.keys())[0]]['dim'], name))
        for j in data.keys():
            #j=str(j)
            dim = data[j]['dim']
            R = torch.as_tensor(data[j]['v']) if 'groundtruth' not in data[j].keys() else torch.as_tensor(data[j]['groundtruth']['v'])
            H_gt = (R @ torch.as_tensor(data[j]['svd_basis']) @ R.T).abs().mean(dim=0)
            H_gt[H_gt != torch.clamp(H_gt, clamp_gt)] = 0
            zero_norm_gt = len(H_gt.nonzero())
            sparse.append(zero_norm_gt)
            method = Method.Manifold_Opt if 'h_size' not in data[j].keys() else Method.Manifold_Opt_GS
            if 'h_size' not in data[j].keys():
                H_man_opt_re, H_man_opt_la = compute_hessian_rotmatrix(data[j], method)
                H_man_opt_la[H_man_opt_la != torch.clamp(H_man_opt_la, clamp)] = 0
                zero_norm_H_man_opt_la = len(H_man_opt_la.nonzero())
                if zero_norm_H_man_opt_la-zero_norm_gt > 0:
                    non_zero_man_opt_la.append([j, zero_norm_H_man_opt_la-zero_norm_gt])
                    if zero_norm_H_man_opt_la-zero_norm_gt < 3:
                        man_opt_count['gt']['la'][zero_norm_H_man_opt_la-zero_norm_gt] +=1
                    elif zero_norm_H_man_opt_la-zero_norm_gt >= 3:
                        man_opt_count['gt']['la'][3] += 1
                H_man_opt_re[H_man_opt_re != torch.clamp(H_man_opt_re, clamp)] = 0
                zero_norm_H_man_opt_re = len(H_man_opt_re.nonzero())
                if zero_norm_H_man_opt_re - zero_norm_gt > 0:
                    non_zero_man_opt_re.append([j, zero_norm_H_man_opt_re - zero_norm_gt])
                    if zero_norm_H_man_opt_re-zero_norm_gt < 3 and zero_norm_H_man_opt_re-zero_norm_gt>0:
                        man_opt_count['gt']['re'][zero_norm_H_man_opt_re - zero_norm_gt] += 1
                    elif zero_norm_H_man_opt_re-zero_norm_gt >= 3:
                        man_opt_count['gt']['re'][3] += 1
                H_man_opt_re_noise, H_man_opt_la_noise = compute_hessian_rotmatrix(data[j], method,
                                                                                   noisy_rot=True)
                H_man_opt_la_noise[H_man_opt_la_noise != torch.clamp(H_man_opt_la_noise, clamp_noise)] = 0
                zero_norm_H_man_opt_la_noise = len(H_man_opt_la_noise.nonzero())
                if zero_norm_H_man_opt_la_noise - zero_norm_gt > 0:
                    non_zero_man_opt_la_noise.append([j, zero_norm_H_man_opt_la_noise - zero_norm_gt])
                    if zero_norm_H_man_opt_la_noise-zero_norm_gt < 3 and zero_norm_H_man_opt_la_noise-zero_norm_gt>0:
                        man_opt_count['noise']['la'][zero_norm_H_man_opt_la_noise - zero_norm_gt] += 1
                    elif zero_norm_H_man_opt_la_noise-zero_norm_gt >= 3:
                        man_opt_count['noise']['la'][3] += 1
                H_man_opt_re_noise[H_man_opt_re_noise != torch.clamp(H_man_opt_re_noise, clamp_noise)] = 0
                zero_norm_H_man_opt_re_noise = len(H_man_opt_re_noise.nonzero())
                if zero_norm_H_man_opt_re_noise - zero_norm_gt > 0:
                    non_zero_man_opt_re_noise.append([j, zero_norm_H_man_opt_re_noise - zero_norm_gt])
                    if zero_norm_H_man_opt_re_noise-zero_norm_gt < 3:
                        man_opt_count['noise']['re'][zero_norm_H_man_opt_re_noise - zero_norm_gt] += 1
                    else:
                        man_opt_count['noise']['re'][3] += 1
            else:
                H_man_opt_gs_re, H_man_opt_gs_la = compute_hessian_rotmatrix(data[j], method)
                H_man_opt_gs_la[H_man_opt_gs_la != torch.clamp(H_man_opt_gs_la, clamp)] = 0
                zero_norm_H_man_opt_gs_la = len(H_man_opt_gs_la.nonzero())
                if zero_norm_H_man_opt_gs_la - zero_norm_gt > 0:
                    non_zero_man_opt_gs_la.append([j, zero_norm_H_man_opt_gs_la - zero_norm_gt])
                    if zero_norm_H_man_opt_gs_la - zero_norm_gt < 3:
                        man_opt_gs_count['gt']['la'][zero_norm_H_man_opt_gs_la - zero_norm_gt] += 1
                    else:
                        man_opt_gs_count['gt']['la'][3] += 1
                H_man_opt_gs_re[H_man_opt_gs_re != torch.clamp(H_man_opt_gs_re, clamp)] = 0
                zero_norm_H_man_opt_gs_re = len(H_man_opt_gs_re.nonzero())
                if zero_norm_H_man_opt_gs_re - zero_norm_gt > 0:
                    non_zero_man_opt_gs_re.append([j, zero_norm_H_man_opt_gs_re - zero_norm_gt])
                    if zero_norm_H_man_opt_gs_re - zero_norm_gt < 3:
                        man_opt_gs_count['gt']['re'][zero_norm_H_man_opt_gs_re - zero_norm_gt] += 1
                    else:
                        man_opt_gs_count['gt']['re'][3] += 1
                H_man_opt_gs_re_noise, H_man_opt_gs_la_noise = compute_hessian_rotmatrix(data[j], method, noisy_rot=True)
                H_man_opt_gs_la_noise[H_man_opt_gs_la_noise != torch.clamp(H_man_opt_gs_la_noise, clamp_noise)] = 0
                zero_norm_H_man_opt_gs_la_noise = len(H_man_opt_gs_la_noise.nonzero())
                if zero_norm_H_man_opt_gs_la_noise - zero_norm_gt > 0:
                    non_zero_man_opt_gs_la_noise.append([j, zero_norm_H_man_opt_gs_la_noise - zero_norm_gt])
                    if zero_norm_H_man_opt_gs_la_noise - zero_norm_gt < 3:
                        man_opt_gs_count['noise']['la'][zero_norm_H_man_opt_gs_la_noise - zero_norm_gt] += 1
                    else:
                        man_opt_gs_count['noise']['la'][3] += 1
                H_man_opt_gs_re_noise[H_man_opt_gs_re_noise != torch.clamp(H_man_opt_gs_re_noise, clamp_noise)] = 0
                zero_norm_H_man_opt_gs_re_noise = len(H_man_opt_gs_re_noise.nonzero())
                if zero_norm_H_man_opt_gs_re_noise - zero_norm_gt > 0:
                    non_zero_man_opt_gs_re_noise.append([j, zero_norm_H_man_opt_gs_re_noise - zero_norm_gt])
                    if zero_norm_H_man_opt_gs_re_noise - zero_norm_gt < 3:
                        man_opt_gs_count['noise']['re'][zero_norm_H_man_opt_gs_re_noise - zero_norm_gt] += 1
                    else:
                        man_opt_gs_count['noise']['re'][3] += 1

        print('\n Landing ')
        if 'h_size' not in data[j].keys():
            print('\nMan_OPT: ')
            print(len(non_zero_man_opt_la), non_zero_man_opt_la, '\n')
            print(len(non_zero_man_opt_la_noise), non_zero_man_opt_la_noise, '\n')
        else:
            print('\nMan_OPT_GS')
            print(len(non_zero_man_opt_gs_la), non_zero_man_opt_gs_la, '\n')
            print(len(non_zero_man_opt_gs_la_noise), non_zero_man_opt_gs_la_noise, '\n')
        print('\n Retraction')
        if 'h_size' not in data[j].keys():
            print('\nMan_OPT: ')
            print(len(non_zero_man_opt_re), non_zero_man_opt_re, '\n')
            print(len(non_zero_man_opt_re_noise), non_zero_man_opt_re_noise, '\n')
            print(man_opt_count, '\n')
        else:
            print('\nMan_OPT_GS')
            print(len(non_zero_man_opt_gs_re), non_zero_man_opt_gs_re, '\n')
            print(len(non_zero_man_opt_gs_re_noise), non_zero_man_opt_gs_re_noise, '\n')
            print(man_opt_gs_count, '\n')


