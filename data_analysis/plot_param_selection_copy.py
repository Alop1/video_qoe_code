# -*- coding: utf-8 -*-
import pandas as pd
import os, re
import matplotlib.pyplot as plt

LOG_DIR = "reports/SELECT_PARAMS"
RESERCHED_SETS = [6, 24, 7]
PARSED_DATA_ALL_SETS = {}

for ele in RESERCHED_SETS:
    PARSED_DATA_ALL_SETS[ele] = []

def main():
    # read CSV
    plot_vector = []
    for file in os.listdir(LOG_DIR):
        # print file
        df_all_executions = pd.read_csv(LOG_DIR + "/" + file, skipinitialspace=True)
        models_names = []
        sorted_r2_for_file = {}
        for model_name, df_region in df_all_executions.groupby('model'):
            sorted_r2_per_model = []
            models_names.append(model_name)
            r2_list_per_featureset = []
            feature_set_list = []
            df_mode_params = df_region["model_parameter"]
            try:
                single_model_param =  df_mode_params.iloc[2]
            except IndexError:
                continue
            param_info = get_model_param(single_model_param)
            # print param_info
            for feature_set, df_subregion in df_region.groupby('features'):
                df_subregion = df_subregion["r_2"]
                r2_list = [float(ele) for ele in df_subregion]

                r2_mean = sum(r2_list) / len(r2_list)

                r2_list_per_featureset.append(r2_mean)

                feature_set_list.append(str(feature_set))

                sorted_r2_per_model.append((r2_mean, feature_set, param_info))  # WANzny albo to albo  stdev

                # if "SVR" in model_name:
                #     print r2_mean, feature_set, param_info

            sorted_r2_for_file[model_name] = sorted_r2_per_model  # tu sa szunkane wartosci



        for model in sorted_r2_for_file:

            for set_number in RESERCHED_SETS:
                selected_dataset = sorted_r2_for_file[model][set_number]  # 0 - aggregate vmaf
                digit_pattern = r".*_(\d+)"
                exec_number = int(re.match(digit_pattern, file).group(1))

                input = (99999, model, selected_dataset[1], selected_dataset[0], selected_dataset[-1])    # dane z jedego zestawu na jeden plik
                PARSED_DATA_ALL_SETS[set_number].append(input)
                plot_vector.append(input)
                # print input
    print "----------------------------------------------------------------------\n\n"

    plot_vector.sort(key=lambda x: x[-1][0])        #sortowanie to moze byc zmienne
    # print plot_vector

    # for item in plot_vector



    # --------------------------------------------------------
    # CONST LIST
    rf_const_tree_n = 2            # 2,6,8,10
    rf_const_depth = 10            # 10,20,40, 60, 70, 100
    svr_const_c_param = 0.7        # 0.7, 10, 20, 30, 40, 60
    svr_const_e_param = 0.1        # 0.1, 0.3, 0.4, 0.5
    nn_const_hidden_l_n = 4        # 1,2,3,4
    nn_const_neurons_n = 5         # 2,3,5,9,13,15
    # --------------------------------------------------------

    CURR_CONST = rf_const_depth
    CONST_RF_LIST = [2, 6, 8, 10]
    CONST_NN_LIST = [1, 2, 3, 4]
    CONST_SVR_LIST = [0.1, 0.3, 0.4, 0.5]


    RF_sum = {}
    for data_set in RESERCHED_SETS:
        RF_sum[data_set] = {}
        for const in CONST_RF_LIST:
            RF_sum[data_set][const] = ([], [])

    NN_sum = {}
    for data_set in RESERCHED_SETS:
        NN_sum[data_set] = {}
        for const in CONST_NN_LIST:
            NN_sum[data_set][const] = ([], [])

    SVR_sum = {}
    for data_set in RESERCHED_SETS:
        SVR_sum[data_set] = {}
        for const in CONST_SVR_LIST:
            SVR_sum[data_set][const] = ([], [])
    print NN_sum

    charts = []

    for id, (dataset_number, data)  in enumerate(PARSED_DATA_ALL_SETS.items()):
        print "set n:  ",dataset_number, " ->"
        # print  data

        data.sort(key=lambda x: x[-1][-1]) #posortowane wg tej drugiemj wartosci z para,metrow
        for input in data:
            # print input
            model_name = input[1]
            data_set = input[2]
            r2_val = input[3]
            para_val = input[4]
            const_val = para_val[0]
            # print "model ", model_name
            # print "dataset ", data_set
            # print "r2 ", r2_val
            # print "param ", para_val
            # print "const_param ", const_val
            # print "\n"

            if "RandomForestRegressor" in model_name:
                # print RF_sum[dataset_number]
                RF_sum[dataset_number][const_val][0].append(r2_val)
                RF_sum[dataset_number][const_val][1].append(para_val[1])

            if "MLPRegressor" in model_name:
                # print NN_sum[dataset_number]
                NN_sum[dataset_number][const_val][0].append(r2_val)
                NN_sum[dataset_number][const_val][1].append(para_val[1])

            if "SVR" in model_name:
                print input
                print SVR_sum[dataset_number]
                print "\n"
                SVR_sum[dataset_number][const_val][0].append(r2_val)
                SVR_sum[dataset_number][const_val][1].append(para_val[1])




        # charts.append(plt.plot(params_nn, r2_nn, '-o', c=color_list[id], label="RF2_zestaw_"+str(dataset_number)))
        # charts.append(plt.plot(params_rf, r2_rf, '-+', c=color_list[id], label="RF_zestaw_" + str(dataset_number)))
        # charts.append(plt.plot(params_svr, r2_svr, '-x', c=color_list[id], label="SVR_zestaw_" + str(dataset_number)))



    color_list = ['r', 'b', 'y', 'g']
    green_colors = ['forestgreen', "limegreen", "mediumseagreen", "palegreen"]
    blue_colors = ["navy", "mediumblue", "dodgerblue", "skyblue"]
    red_colors = ["maroon", "brown", "tomato", "lightsalmon"]

    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # ------------------------------------------------------
    const_name = u" głębokość drzew-"
    # const_name = u" liczba ukrytych warst-"
    # const_name = u" epsilon-"

    # x_label = "liczba drzew"
    # y_label = "R-kwadrat"
    # m_title = "Las losowy"

    x_label = "parametr C"
    y_label = "R-kwadrat"
    m_title = "SVR"

    # x_label = u"liczba neuronów w warstwie ukrytej"
    # y_label = "R-kwadrat"
    # m_title = u"sieć neuronowa"
    # ------------------------------------------------------

    prev_val = ""
    for dataset, vals in SVR_sum.items():
        for  (const, val) in vals.items():
            id_to_rm = []
            for id, var_val in enumerate(val[1]):
                print "curr: ", var_val, "prev ", prev_val
                if var_val == prev_val or  var_val == 60:
                    id_to_rm.append(id)
                    print "removing"
                    # del val[1][id]
                    # del val[0][id]
                    # print val
                prev_val = var_val
            id_to_rm.sort()
            print id_to_rm[::-1]
            print [val[1][id] for id in id_to_rm]
            for id in id_to_rm[::-1]:
                del val[0][id]
                del val[1][id]


    print "\n\n final: "
    for dataset, vals in SVR_sum.items():
        print "zestaw ", dataset, "-> ", vals
        for const, val in vals.items():
            print "  const ", const, "-> ", val
        print "\n"

    from ordered_set import OrderedSet
    # from orderedset import OrderedSet
    from collections import OrderedDict
    import operator

    for dataset, vals in SVR_sum.items():
        ordered_vals = sorted(vals.items(), key=operator.itemgetter(0))
        print ordered_vals
        ordered_dict = {key: vals for key, vals in ordered_vals}
        print ordered_dict
        for const, val in ordered_vals:
            x_ax = val[1]
            y_ax = val[0]
            if dataset == 24:
                charts.append(ax.plot(x_ax, y_ax, '-o', c=green_colors[-1], label="zestaw_"+ str(dataset) +const_name+str(const)))
                green_colors.pop()
            if dataset == 6:
                charts.append(ax.plot(x_ax, y_ax, '-o', c=blue_colors[-1], label="zestaw_"+ str(dataset) + const_name +str(const)))
                blue_colors.pop()
            if dataset == 7:
                charts.append(ax.plot(x_ax, y_ax, '-o', c=red_colors[-1], label="zestaw_"+ str(dataset) + const_name +str(const)))
                red_colors.pop()


    ax.grid(color='gray', linestyle='-', linewidth=0.1)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(m_title)


    plt.show()


def get_model_param(raw_params):
    if "hidden_layer_sizes" in raw_params:
        param_info = get_NN_params(raw_params)
    elif "max_depth=" in raw_params:
        param_info = get_RF_params(raw_params)
    elif "kernel=" in raw_params:
        param_info = get_SVR_params(raw_params)
    return param_info


def get_NN_params(all_params):
    # all_params = "(activation='relu'| alpha=0.0001| batch_size='auto'| beta_1=0.9|beta_2=0.999| early_stopping=False| epsilon=1e-08|hidden_layer_sizes=(2|)| learning_rate='constant'|learning_rate_init=0.001| max_iter=5000| momentum=0.9|n_iter_no_change=10| nesterovs_momentum=True| power_t=0.5|random_state=None| shuffle=True| solver='adam'| tol=0.0001|validation_fraction=0.1| verbose=False| warm_start=False)"
    # all_params = "(activation='relu'| alpha=0.0001| batch_size='auto'| beta_1=0.9|beta_2=0.999| early_stopping=False| epsilon=1e-08|hidden_layer_sizes=(15| 15| 15)| learning_rate='constant'|learning_rate_init=0.001| max_iter=5000| momentum=0.9|n_iter_no_change=10| nesterovs_momentum=True| power_t=0.5|random_state=None| shuffle=True| solver='adam'| tol=0.0001|validation_fraction=0.1| verbose=False| warm_start=False),['PSNR'| 'Aggregate_vmaf'],0.524558641186,(677| 140),,[]"
    hidden_layer_pattern = r"\(((\d+(\|)?\s?)+)\)"
    hidden_layer_row = re.search(hidden_layer_pattern, all_params).group(1)
    hidden_l_temp = hidden_layer_row.split(" ")
    neuron_no = int(re.search(r"(\d+)", hidden_l_temp[0]).group(1))
    hidden_l_no = len(hidden_l_temp)
    param_info = (hidden_l_no, neuron_no )
    # print param_info
    return param_info

def get_RF_params(all_params):
    # all_params = "RandomForestRegressor,(bootstrap=True| criterion='mse'| max_depth=8|   max_features='auto'| max_leaf_nodes=None|   min_impurity_decrease=0.0| min_impurity_split=None|   min_samples_leaf=1| min_samples_split=2|   min_weight_fraction_leaf=0.0| n_estimators=100| n_jobs=None|   oob_score=False| random_state=None| verbose=0| warm_start=False),['PSNR'| 'Aggregate_vmaf'],0.565899864863,(677| 140),,[]"
    depth = int(re.search(r"max_depth=(\d+)", all_params).group(1))
    n_estimators = int(re.search(r"n_estimators=(\d+)", all_params).group(1))
    return (depth, n_estimators)

def get_SVR_params(all_params):
    # all_params = "SVR,(C=20| cache_size=200| coef0=0.0| degree=3| epsilon=0.5|  gamma='auto_deprecated'| kernel='rbf'| max_iter=-1| shrinking=True|  tol=0.001| verbose=False),['PSNR'| 'Aggregate_vmaf'| 'SSIM'| 'MS_SSIM'| 'duration'| 'one_res'| '1920x1080'| '352x288'| '3840x2160'| '640x480'| '704x576'],0.636113273139,(677| 140),,[]"
    c_param = float(re.search(r".*C=(\d*\.?\d*)", all_params).group(1))
    e_param = float(re.search(r"epsilon=(\d*\.?\d*)", all_params).group(1))
    param_info = (e_param, c_param)
    print param_info
    return param_info


if __name__ == '__main__':
    main()


    # TEST
    # print get_NN_params("e")
    # print get_SVR_params("e")
    # print get_RF_params("e")