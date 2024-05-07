import json
import torch
from os.path import dirname, abspath
out_dir = dirname(dirname(abspath(__file__))) + '/Output_algorithms/Random_matrices/Time_complexity'
report_dir = "/store/steinitz/datastore/fatimaba/Github/trafo_nova/Anova_AE/Test_Cases_Plots"
names_gs = [
    'Time_Complexity_GS_h_1.0_bh_3.14_dim_2_gs.json',
    'Time_Complexity_GS_h_0.5_bh_3.14_dim_2_gs.json',
    'Time_Complexity_GS_h_0.25_bh_3.14_dim_2_gs.json',
    'Time_Complexity_GS_h_0.125_bh_3.14_dim_2_gs.json',
    'Time_Complexity_GS_h_0.1_bh_3.14_dim_2_gs.json',

    'Time_Complexity_GS_h_1.0_bh_1.57_dim_3_gs.json',
    'Time_Complexity_GS_h_0.5_bh_1.57_dim_3_gs.json',
    'Time_Complexity_GS_h_0.25_bh_1.57_dim_3_gs.json',
    'Time_Complexity_GS_h_0.125_bh_1.57_dim_3_gs.json',
    'Time_Complexity_GS_h_0.1_bh_1.57_dim_3_gs.json',

    'Time_Complexity_GS_h_1.0_bh_1.57_dim_4_gs.json',
    'Time_Complexity_GS_h_0.5_bh_1.57_dim_4_gs.json',
    'Time_Complexity_GS_h_0.25_bh_1.57_dim_4_gs.json',
    'Time_Complexity_GS_h_0.125_bh_1.57_dim_4_gs.json',
    'Time_Complexity_GS_h_0.1_bh_1.00_dim_4_gs.json',

    'Time_Complexity_GS_h_1.0_bh_2.36_dim_5_gs.json']

name_ri = ['Time_Complexity_RI_dim_2_gs.json',
           'Time_Complexity_RI_dim_3_gs.json',
           'Time_Complexity_RI_dim_4_gs.json',
           'Time_Complexity_RI_dim_5_gs.json',
           ]

def evaluate_runtime(c_files):
    man_opt_GS = False if c_files == names_gs else True
    for name in c_files:
        print("\n Filename: ", name)
        d = 0
        with open('{}/{}'.format(out_dir, name), 'r') as convert_file:
            data = json.load(convert_file)
            rot_times = torch.zeros(len(list(data.keys())))
            loss_times = torch.zeros(len(list(data.keys())))
            if man_opt_GS:
                man_opt = torch.zeros((2, len(list(data.keys()))))
                for index_j, j in enumerate(data.keys()):
                    d = data[j]['d']
                    man_opt[0, index_j] = data[j]['time']['clean']['Man_Opt_RI']['rgd'][0]
                    man_opt[1, index_j] = data[j]['time']['clean']['Man_Opt_RI']['la'][0]
            else:
                for index_j, j in enumerate(data.keys()):
                    rot_times[index_j] = data[str(j)]['time']['clean']['Grid_search'][0]
                    loss_times[index_j] = data[str(j)]['time']['clean']['Grid_search'][1] / data[j]['hessian_rank'][
                        'clean']
        if not man_opt_GS:
            print("\n Mean rotation time complexity: ", rot_times.mean())
            print("\n Mean loss time complexity: ", loss_times.mean(), loss_times.mean() * (d * (d + 1) / 2))
        else:
            print("\n Mean time complexity Rgd RI: ", man_opt[0, :].mean())
            print("\n Mean time complexity LA RI: ", man_opt[1, :].mean())

evaluate_runtime(name_ri)
evaluate_runtime(names_gs)


