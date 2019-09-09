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
        print file
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
            print param_info
            for feature_set, df_subregion in df_region.groupby('features'):
                df_subregion = df_subregion["r_2"]
                r2_list = [float(ele) for ele in df_subregion]

                r2_mean = sum(r2_list) / len(r2_list)

                r2_list_per_featureset.append(r2_mean)

                feature_set_list.append(str(feature_set))

                sorted_r2_per_model.append((r2_mean, feature_set, param_info))  # WANzny albo to albo  stdev

            sorted_r2_for_file[model_name] = sorted_r2_per_model  # tu sa szunkane wartosci



        for model in sorted_r2_for_file:

            for set_number in RESERCHED_SETS:
                selected_dataset = sorted_r2_for_file[model][set_number]  # 0 - aggregate vmaf
                digit_pattern = r".*_(\d+)"
                exec_number = int(re.match(digit_pattern, file).group(1))

                input = (exec_number, model, selected_dataset[1], selected_dataset[0], selected_dataset[-1])    # dane z jedego zestawu na jeden plik
                PARSED_DATA_ALL_SETS[set_number].append(input)
                plot_vector.append(input)
                # print input


    plot_vector.sort(key=lambda x: x[-1][0])        #sortowanie to moze byc zmienne
    # print plot_vector

    # for item in plot_vector


    color_list = ['r', 'b', 'y', 'g']
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
    CONST_RF_LIST = [2,6,8,10]


    charts = []
    draw_vactors = {}
    RF_charts = {}
    i = 0
    j = 0
    for id, (dataset_number, data)  in enumerate(PARSED_DATA_ALL_SETS.items()):
        i += 1
        params_nn = []
        r2_nn = []
        params_rf = []
        r2_rf = []
        params_svr = []
        r2_svr = []

        data.sort(key=lambda x: x[-1][0])
        for input in data:
            j += 1
            model_name = input[1]
            sim_n =input[0]
            data_set = input[2]
            r2_val = input[3]
            para_val = input[4]

            # if "MLPRegressor" in model_name and nn_const_hidden_l_n in para_val:
            #     print input
            #     hidden_l_n = para_val[0]
            #     neurons_n = para_val[1]
            #     params_nn.append(neurons_n)
            #     r2_nn.append(r2_val)

            if "SVR" in model_name and svr_const_c_param in para_val:
                print input
                c_param = para_val[0]
                e_param = para_val[1]
                params_rf.append(e_param)
                r2_rf.append(r2_val)


            if "RandomForestRegressor" in model_name:
                for const in CONST_RF_LIST:
                    if const == para_val[0]:
                        # print input
                        depth = para_val[0]
                        tree_n = para_val[1]
                        params_rf.append(tree_n)
                        r2_rf.append(r2_val)
                        x_label = "liczba drzew"
                        y_label = "R-kwadrat"
                        m_title = "Las losowy"
                        if const not in RF_charts:
                            RF_charts[const] = ([tree_n], [r2_val])
                        else:
                            # print "rf chart ", RF_charts
                            RF_charts[const][0].append(tree_n)
                            RF_charts[const][1].append(r2_val)
                            print "iter i: ", i, "j: ", j
                            print RF_charts
                            print "\n"



            # if "RandomForestRegressor" in model_name and rf_const_tree_n == para_val[0]:
            #     print input
            #     depth = para_val[0]
            #     tree_n = para_val[1]
            #     params_nn.append(tree_n)
            #     r2_nn.append(r2_val)
            #     x_label = "liczba drzew"
            #     y_label = "R-kwadrat"
            #     m_title = "Las losowy"

        # draw_vactors[dataset_number] = params
        # charts.append(plt.plot(params_nn, r2_nn, '-o', c=color_list[id], label="RF2_zestaw_"+str(dataset_number)))
        # charts.append(plt.plot(params_rf, r2_rf, '-+', c=color_list[id], label="RF_zestaw_" + str(dataset_number)))
        # charts.append(plt.plot(params_svr, r2_svr, '-x', c=color_list[id], label="SVR_zestaw_" + str(dataset_number)))

        # for draw_vector in RF_charts:
        #     charts.append(plt.plot(draw_vector[0], draw_vector[1], '-+', c=color_list[id], label="zestaw_" + str(dataset_number)))
        #     plt.annotate(str(draw_vector[-1]),  # this is the text
        #                  (draw_vector[0][-1], draw_vector[0][-1]),  # this is the point to label
        #                  textcoords="offset points",  # how to position the text
        #                  xytext=(0, 10),  # distance from text to points (x,y)
        #                  ha='center')  # horizontal alignment can be left, right or center
        #     break


        for key, val  in RF_charts.items():
            print "vect ", key, " ", val

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(m_title)





    plt.legend()

    plt.show()

    #--------------------------------------------------------------------------------
    # sim_number = [ele[0] for ele in plot_vector if "MLPRegressor" in ele[1]]
    # r2_list_nn = [ele[-1] for ele in plot_vector if "MLPRegressor" in ele[1]]
    #
    # sim_number = [ele[0] for ele in plot_vector if "SVR" in ele[1]]
    # r2_list_svr = [ele[-1] for ele in plot_vector if "SVR" in ele[1]]
    #
    # sim_number = [ele[0] for ele in plot_vector if "RandomForestRegressor" in ele[1]]
    # r2_list_rf = [ele[-1] for ele in plot_vector if "RandomForestRegressor" in ele[1]]
    #
    # print r2_list
    # models_names = ['Regresja Liniowa', u'Sieć neuronowa', 'Las Losowy', 'SVR']
    # charts = []
    # # charts.append(plt.scatter(sim_number, r2_list_linear, c='r'))
    # # charts.append(plt.scatter(sim_number, r2_list_nn, c='b'))
    # # charts.append(plt.scatter(sim_number, r2_list_rf, c='y'))
    # # charts.append(plt.scatter(sim_number, r2_list_svr, c='g'))
    #
    # charts.append(plt.plot(sim_number, r2_list_linear, '-o', c='r', label='Regresja Liniowa'))
    # charts.append(plt.plot(sim_number, r2_list_nn, '-o', c='b', label=u'Sieć neuronowa'))
    # charts.append(plt.plot(sim_number, r2_list_rf, '-o', c='y', label='Las Losowy'))
    # charts.append(plt.plot(sim_number, r2_list_svr, '-o', c='g', label='SVR'))
    #
    # # plt.legend(charts, models_names)
    #
    # plt.legend()
    #
    # plt.show()

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
    param_info = (c_param, e_param)
    # print param_info
    return param_info


if __name__ == '__main__':
    main()


    # TEST
    # print get_NN_params("e")
    # print get_SVR_params("e")
    # print get_RF_params("e")