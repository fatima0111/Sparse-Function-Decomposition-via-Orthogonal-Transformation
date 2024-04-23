import json
import torch
from Utils.Function_utils import compute_hessian_orig_2d, compute_gradient_orig_2d
from Anova_AE.Libs.Evaluation_utils import get_total_Rot

if __name__ == '__main__':
    clamp = 6e-4
    out_dir = "/store/steinitz/datastore/fatimaba/Github/trafo_nova/Anova_AE/Output_ManOpt/Comparing_Man_Opt_GS"
    report_dir = "/store/steinitz/datastore/fatimaba/Github/trafo_nova/Anova_AE/Test_Cases_Plots"
    names_clean = [
        'function_Man_opt_grid_search_h_1.0_N_50.json',
        'function_Man_opt_grid_search_h_0.5_N_50.json',
        'function_Man_opt_grid_search_h_0.25_N_50.json',
        'function_Man_opt_grid_search_h_0.125_N_50.json'
        ]
    names_noisy = [
        'function_Man_opt_grid_search_cov_0.5_h_1.0_N_50.json',
        'function_Man_opt_grid_search_cov_0.5_h_0.5_N_50.json',
        'function_Man_opt_grid_search_cov_0.5_h_0.25_N_50.json',
        'function_Man_opt_grid_search_cov_0.5_h_0.125_N_50.json'
    ]
    #Non_zeros = [[], [], [], []]
    list_false_blocks = []
    print_string = ''
    for name in names:
        GNorm_diff_mo_la = []
        GNorm_diff_mo_re = []

        HNorm_diff_mo_la = []
        HNorm_diff_mo_re = []

        GNorm_diff_la = []
        GNorm_diff_re = []

        HNorm_diff_la = []
        HNorm_diff_re = []

        num_false_block = 0
        list_false_block = []
        with open('{}/{}'.format(
                out_dir, name), 'r') as convert_file:
            data = json.load(convert_file)
            print_string += '\n name: {} \n'.format(name)
            print('\n name: {} \n'.format(name))
            for j in data.keys():
                zero_norm_H_la = 0
                zero_norm_H_re = 0
                zero_norm_H_la_noise = 0
                zero_norm_H_re_noise = 0
                ground_truth = data[j]['groundtruth']
                ground_truth['v'] = torch.as_tensor(ground_truth['v'])
                dim = data[j]['dim']
                Blocks = ground_truth['Blocs']
                BlockSizes = data[j]['SBD_GT']

                v = ground_truth['v']
                x_test = data[j]['x_test'] = torch.as_tensor(data[j]['x_test'])
                supp = int(data[j]['support'])
                h_size = float(name.split('_h_')[1].split('_N')[0])
                gradF_orig = compute_gradient_orig_2d(
                    x_test.clone(), ground_truth)
                hessianF_orig = compute_hessian_orig_2d(
                    x_test.clone(), ground_truth)

                G_gt = (v @ gradF_orig).abs().mean(dim=1)
                G_gt[G_gt != torch.clamp(G_gt, clamp)] = 0
                zero_norm_gt = len(G_gt.nonzero())

                H_gt = ((v @ hessianF_orig @ v.T).abs()**2).mean(dim=0)**.5
                H_gt[H_gt != torch.clamp(H_gt, clamp)] = 0
                zero_norm_ht = len(H_gt.nonzero())
                U1 = torch.as_tensor(data[j]['grad_U'])

                Rot_la, Rot_re = get_total_Rot(data, j)
                if h_size == 1/2:
                    Rot_mo_la, Rot_mo_re = get_total_Rot(data, j, man_opt=True)
                G_la = (Rot_la @ gradF_orig).abs().mean(dim=1)

                G_la[G_la != torch.clamp(G_la, clamp)] = 0
                zero_norm_G_la = len(G_la.nonzero())

                G_re = (Rot_re @ gradF_orig).abs().mean(dim=1)
                G_re[G_re != torch.clamp(G_re, clamp)] = 0
                zero_norm_G_re = len(G_re.nonzero())

                H_la = ((Rot_la@hessianF_orig@Rot_la.T).abs()**2).mean(dim=0)**.5
                H_la[H_la != torch.clamp(H_la, clamp)] = 0
                zero_norm_H_la = len(H_la.nonzero())
                if len(data[j]['SBD_GT']) == 1:
                    num_false_block += 1
                    list_false_block.append(int(j))

                H_re = ((Rot_re@hessianF_orig@Rot_re.T).abs()**2).mean(dim=0)**.5
                H_re[H_re != torch.clamp(H_re, clamp)] = 0
                zero_norm_H_re = len(H_re.nonzero())
                if zero_norm_G_la - zero_norm_gt > 0:

                    GNorm_diff_la.append([j, zero_norm_G_la - zero_norm_gt])
                if zero_norm_G_re - zero_norm_gt > 0:
                    GNorm_diff_re.append([j, zero_norm_G_re - zero_norm_gt])
                if zero_norm_H_la - zero_norm_ht > 0:
                    HNorm_diff_la.append([j,  zero_norm_H_la - zero_norm_ht])
                if zero_norm_H_re - zero_norm_ht > 0:
                    HNorm_diff_re.append([j, zero_norm_H_re - zero_norm_ht])
                if h_size == 1/2:
                    G_mo_la = (Rot_mo_la @ gradF_orig).abs().mean(dim=1)

                    G_mo_la[G_mo_la != torch.clamp(G_mo_la, clamp)] = 0
                    zero_norm_G_mo_la = len(G_mo_la.nonzero())

                    G_mo_re = (Rot_mo_re @ gradF_orig).abs().mean(dim=1)
                    G_mo_re[G_mo_re != torch.clamp(G_mo_re, clamp)] = 0
                    zero_norm_G_mo_re = len(G_mo_re.nonzero())

                    H_mo_la = ((Rot_mo_la @ hessianF_orig @ Rot_la.T).abs()**2).mean(dim=0)**.5
                    H_mo_la[H_mo_la != torch.clamp(H_mo_la, clamp)] = 0
                    zero_norm_H_mo_la = len(H_mo_la.nonzero())

                    H_mo_re = ((Rot_mo_re @ hessianF_orig @ Rot_mo_re.T).abs()**2).mean(dim=0)**.5
                    H_mo_re[H_mo_re != torch.clamp(H_mo_re, clamp)] = 0
                    zero_norm_H_mo_re = len(H_mo_re.nonzero())
                    if zero_norm_G_mo_la - zero_norm_gt > 0:
                        GNorm_diff_mo_la.append([j, zero_norm_G_mo_la - zero_norm_gt])
                    if zero_norm_G_mo_re - zero_norm_gt > 0:
                        GNorm_diff_mo_re.append([j, zero_norm_G_mo_re - zero_norm_gt])
                    if zero_norm_H_mo_la - zero_norm_ht > 0:
                        HNorm_diff_mo_la.append([j, zero_norm_H_mo_la - zero_norm_ht])
                    if zero_norm_H_mo_re - zero_norm_ht > 0:
                        HNorm_diff_mo_re.append([j, zero_norm_H_mo_re - zero_norm_ht])
            list_false_blocks.append(list_false_block)
            print('\n num_false_block: ', num_false_block)
        if h_size == 1/2:
            print_string += '\n Landing MO \n'
            print('\n Landing MO', '\n')
            print_string += 'Grad: {} \n'.format(GNorm_diff_mo_la)
            print('Grad: {} \n'.format(GNorm_diff_mo_la))
            print_string += 'Hess: {} \n'.format(HNorm_diff_mo_la)
            print('Hess: {} \n'.format(HNorm_diff_mo_la))

            print_string += '\n Retraction MO \n'
            print('\n Retraction MO \n')
            print_string += 'Grad: {} \n'.format(GNorm_diff_mo_re)
            print('Grad: {} \n'.format(GNorm_diff_mo_re))
            print_string += 'Hess: {} \n'.format(HNorm_diff_mo_re)
            print('Hess: {} \n'.format(HNorm_diff_mo_re))

        print_string += '\n Landing MO_GS \n'
        print('\n Landing MO_GS \n')
        print_string += 'Grad: {} \n'.format(GNorm_diff_la)
        print('Grad: {} \n'.format(GNorm_diff_la))
        print_string += 'Hess: {} \n'.format(HNorm_diff_la)
        print('Hess: {} \n'.format(HNorm_diff_la))

        print_string += '\n Rgd_grid_search h= \n'.format(h_size)
        print('\n Rgd_grid_search h= \n'.format(h_size))
        print_string += 'Grad: {} \n'.format(GNorm_diff_re)
        print('Grad: {} \n'.format(GNorm_diff_re))
        print_string += 'Hess: {} \n'.format(HNorm_diff_re)
        print('Hess: {} \n'.format(HNorm_diff_re))
    print(torch.as_tensor(list_false_blocks), torch.as_tensor(list_false_blocks).shape)