# -*- coding: utf-8 -*-
import pandas as pd
import os, re
import matplotlib.pyplot as plt


LOG_DIR = "reports/check_sim_quantity"
def main():
    # read CSV
    plot_vector = []
    plot_vector_2 = []
    plot_vector_3 = []
    for file in os.listdir(LOG_DIR):
        df_all_executions = pd.read_csv(LOG_DIR + "/" + file, skipinitialspace=True)
        # print df_all_executions.head(5)
        glob_feature_set_list = []
        # split based on models
        r2_all_models = []
        models_names = []
        sorted_r2_for_file = {}
        for model_name, df_region in df_all_executions.groupby('model'):
            sorted_r2_per_model = []
            models_names.append(model_name)
            r2_list_per_featureset = []
            feature_set_list = []
            # print "df_region type: ", type(df_region)
            # print df_region.head(5)
            counter = 0
            for feature_set, df_subregion in df_region.groupby('features'):
                counter += 1
                df_subregion = df_subregion["r_2"]
                # print "r2_list: ",df_subregion.head(3)
                r2_list = [float(ele) for ele in df_subregion]
                ###### statistic deviation ############
                # st_dev = statistics.stdev(r2_list)
                # print st_dev
                #######################################3
                r2_mean = sum(r2_list)/len(r2_list)
                r2_list_per_featureset.append(r2_mean)
                # r2_list_per_featureset.append(st_dev)     # in case of STDEV

                feature_set_list.append(str(feature_set))
                # print "dataset: {} \nmean r2: {} for model {}, feature {}\n".format(counter, r2_mean, model_name, feature_set)
                sorted_r2_per_model.append((r2_mean,feature_set)) # WANzny albo to albo  stdev
                # sorted_r2_per_model.append((st_dev, feature_set))
            # sorted_r2_per_model.sort(key=lambda x: x[0])
            sorted_r2_for_file[model_name] = sorted_r2_per_model # tu sa szunkane wartosci
            # print "model_name: {}, feature_set_list: {}".format(model_name, feature_set_list)
            # print "model_name {} ile featuresetow{}, ile r2 {}".format(model_name, len(feature_set_list),len(r2_list_per_featureset))
            x_pos = [i for i, _ in enumerate(feature_set_list)]

            glob_feature_set_list = feature_set_list[:]
            r2_all_models.append(r2_list_per_featureset)

        for model in sorted_r2_for_file:
            selected_dataset =  sorted_r2_for_file[model][18]      #0 - aggregate vmaf
            selected_dataset_2 = sorted_r2_for_file[model][7]  # 7 - PSNR, VMAF, SSIM, MS-SSIM, blokowość, aktywność przestrzenna, pillarbox, \\ straty bloków, rozmycie, aktywność czasowa, wyciemnienie, ekspozycja, kontrast, \\ jasność, czas trwania, rozdzielczości
            selected_dataset_3 = sorted_r2_for_file[model][25]  # 25-  blokowość, aktywność przestrzenna, pillarbox, straty bloków, rozmycie, aktywność czasowa,\\ wyciemnienie, ekspozycja,  kontrast, jasność, czas trwania, rozdzielczości, liczba dostępnych

            digit_pattern = r".*_(\d+)"
            exec_number = int(re.match(digit_pattern, file).group(1))
            input = (exec_number, model, selected_dataset[1], selected_dataset[0])
            plot_vector.append(input)

            digit_pattern = r".*_(\d+)"
            exec_number_2 = int(re.match(digit_pattern, file).group(1))
            input_2 = (exec_number_2, model, selected_dataset_2[1], selected_dataset_2[0])
            plot_vector_2.append(input_2)

            digit_pattern = r".*_(\d+)"
            exec_number_3 = int(re.match(digit_pattern, file).group(1))
            input_3 = (exec_number_3, model, selected_dataset_3[1], selected_dataset_3[0])
            plot_vector_3.append(input_3)


    print plot_vector

    plot_vector.sort(key=lambda x: x[0])
    plot_vector_2.sort(key=lambda x: x[0])
    plot_vector_3.sort(key=lambda x: x[0])
    print plot_vector

    sim_number = [ele[0] for ele in plot_vector if "Linear" in ele[1]]
    r2_list_linear  = [ele[-1] for ele in plot_vector if "Linear" in ele[1]]
    r2_list_linear_2 = [ele[-1] for ele in plot_vector_2 if "Linear" in ele[1]]
    r2_list_linear_3 = [ele[-1] for ele in plot_vector_3 if "Linear" in ele[1]]
    print r2_list_linear

    sim_number = [ele[0] for ele in plot_vector if "MLPRegressor" in ele[1]]
    r2_list_nn  = [ele[-1] for ele in plot_vector if "MLPRegressor" in ele[1]]
    r2_list_nn_2 = [ele[-1] for ele in plot_vector_2 if "MLPRegressor" in ele[1]]
    r2_list_nn_3 = [ele[-1] for ele in plot_vector_3 if "MLPRegressor" in ele[1]]

    sim_number = [ele[0] for ele in plot_vector if "SVR" in ele[1]]
    r2_list_svr = [ele[-1] for ele in plot_vector if "SVR" in ele[1]]
    r2_list_svr_2 = [ele[-1] for ele in plot_vector_2 if "SVR" in ele[1]]
    r2_list_svr_3 = [ele[-1] for ele in plot_vector_3 if "SVR" in ele[1]]

    sim_number = [ele[0] for ele in plot_vector if "RandomForestRegressor" in ele[1]]
    r2_list_rf  = [ele[-1] for ele in plot_vector if "RandomForestRegressor" in ele[1]]
    r2_list_rf_2  = [ele[-1] for ele in plot_vector_2 if "RandomForestRegressor" in ele[1]]
    r2_list_rf_3  = [ele[-1] for ele in plot_vector_3 if "RandomForestRegressor" in ele[1]]



    # print r2_list
    models_names = ['Regresja Liniowa', u'Sieć neuronowa', 'Las Losowy', 'SVR']
    charts = []
    # charts.append(plt.scatter(sim_number, r2_list_linear, c='r'))
    # charts.append(plt.scatter(sim_number, r2_list_nn, c='b'))
    # charts.append(plt.scatter(sim_number, r2_list_rf, c='y'))
    # charts.append(plt.scatter(sim_number, r2_list_svr, c='g'))

    charts.append(plt.plot(sim_number, r2_list_linear,  '-o',  c='r', label='Regresja liniowa' ))
    charts.append(plt.plot(sim_number, r2_list_linear_2,  '-o',  c='r',  ))
    charts.append(plt.plot(sim_number, r2_list_linear_2,  '-o',  c='r',  ))

    charts.append(plt.plot(sim_number, r2_list_nn,  '-o',  c='b', label=u'Sieć neuronowa'))
    charts.append(plt.plot(sim_number, r2_list_nn_2,  '-o',  c='b'))
    charts.append(plt.plot(sim_number, r2_list_nn_3,  '-o',  c='b'))

    charts.append(plt.plot(sim_number, r2_list_rf,  '-o',  c='y', label='Las losowy'))
    charts.append(plt.plot(sim_number, r2_list_rf_2,  '-o',  c='y'))
    charts.append(plt.plot(sim_number, r2_list_rf_3,  '-o',  c='y'))

    charts.append(plt.plot(sim_number, r2_list_svr,  '-o',  c='g', label='SVR'))
    charts.append(plt.plot(sim_number, r2_list_svr_2,  '-o',  c='g'))
    charts.append(plt.plot(sim_number, r2_list_svr_3,  '-o',  c='g'))

    # plt.legend(charts, models_names)

    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', framealpha=1)


    plt.show()



if __name__ == '__main__':
    main()