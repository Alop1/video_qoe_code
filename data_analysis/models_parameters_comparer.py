# -*- coding: utf-8 -*-
import pandas as pd
import os
import copy
from matplotlib import pyplot as plt

base_csv_files = []

def read_csv_file(path):
    print ROOT_LOG_PATH + path
    csv_file = pd.read_csv(ROOT_LOG_PATH + path, skipinitialspace=True)
    base_csv_files.append((csv_file, path))


def create_sorted_lists(base_csv_files):
    sorted_lists_all_csv =[]
    for csv_file, path_org in base_csv_files:
        sorted_r2 = {}
        for region, df_region in csv_file.groupby('model'):
            sorted_r2_per_model = []
            # models_names.append(region)
            r2_list_per_featureset = []
            feature_set_list = []
            # print "df_region type: ", type(df_region)
            # print df_region.head(5)
            counter = 0
            for id , (feature_set, df_subregion) in enumerate(df_region.groupby('features')):
                counter += 1
                df_model_parameters = df_subregion["model_parameter"]
                for ele in df_model_parameters:
                    params = str(ele)

                print params
                df_subregion = df_subregion["r_2"]
                # print "r2_list: ", df_subregion.head(3)
                r2_list = [float(ele) for ele in df_subregion]
                r2_mean = sum(r2_list) / len(r2_list)
                r2_list_per_featureset.append(r2_mean)
                feature_set_list.append(str(feature_set))
                # print "dataset: {} \nmean r2: {} for model {}, feature {}\n".format(counter, r2_mean, region, feature_set)
                sorted_r2_per_model.append((r2_mean, feature_set, params, path_org, id+1))
            sorted_r2_per_model.sort(key=lambda x: x[0])
            # print region
            # print sorted_r2_per_model
            sorted_r2[region] = sorted_r2_per_model
            # new_element = copy.deepcopy(sorted_r2)
        sorted_lists_all_csv.append(sorted_r2)
    # for ele in sorted_lists_all_csv:
    #     print ele
    return  sorted_lists_all_csv


def IsExist(model):
    path_to_model = "reports/comparision_models_parameters/"+model[:4]
    if os.path.isfile(path_to_model):
        models_results = open(path_to_model, "a+")
    else:
        models_results = open(path_to_model, "w+")
        header = "r_kwadrat_1, r_kwadrat_2, r_kwadrat_3, model\n"
        models_results.write(header)
    return models_results

def update_csv(csv_file, content, model):
    # print csv_file
    # print content
    best_results = content #content[-3:]
    model_param = best_results[0][-3]
    org_path = best_results[0][-2]
    row_part_1= "**********************************************8\n"
    row_part_1 += str(org_path) + "\n"
    row_part_2 = ""
    for ele in best_results:
        # print ele
        row_part_1 += str(ele[-1]) +". " + str(ele[0]) + "->" + str(ele[1]) + "\n"
        # row_part_2 += str(ele[1]) + ","
    # row_part_1 += "\n"
    # row_part_2 += model + "\n"
    row = row_part_1+row_part_2 + model_param + '\n'
    # print row
    csv_file.write(row)



def create_final_csv(sorted_lists_all_csv):
    for single_csv in sorted_lists_all_csv:
        for model, content in single_csv.items():
            file_per_model = IsExist(model)
            # print content
            # print model
            update_csv(file_per_model, content, model)



ROOT_LOG_PATH = "reports/"
PATH_1 = "only_best_param/sim_0.csv"
# PATH_2 = "czerwiec_3/models_summary_4hl_noresolution_noslice_rbf.csv"
# PATH_3 = "czerwiec_4/models_summary_3hl_noresolution_noslice_rbf.csv"
# ## promoeteusz zle logowanie do pliku PATH_5 = "czerwiec_5/models_summary_smrLinear_svrLinear_4HL_with_res_clean.csv"
# PATH_7 = "czerwiec_7_res/models_summary_4hl_withresolution_noslice_rbf.csv"
# PATH_8 = "main_candidate/models_summary.csv"
# PATH_9 = "czerwiec_7_res/models_summary_FR100_withresolution_noslice_rbf_noduration.csv"
# PATH_10 = "main_candidate_2/models_summary.csv"
# PATH_11 = "main_candidate_2/models_summary_newer.csv"

read_csv_file(PATH_1)
# read_csv_file(PATH_2)
# read_csv_file(PATH_3)
# read_csv_file(PATH_7)
# read_csv_file(PATH_8)
# read_csv_file(PATH_9)
# read_csv_file(PATH_10)
# read_csv_file(PATH_11)
sorted_lists_all_csv = create_sorted_lists(base_csv_files)

create_final_csv(sorted_lists_all_csv)





