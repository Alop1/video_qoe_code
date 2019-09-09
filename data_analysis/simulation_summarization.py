# -*- coding: utf-8 -*-
import pandas as pd
import statistics
from matplotlib import pyplot as plt

LOG_DIR = "reports/only_best_param/"
FILE_NAME = "sim_0.csv"



def main():
    # read CSV
    df_all_executions = pd.read_csv(LOG_DIR + FILE_NAME, skipinitialspace=True)
    # print df_all_executions.head(5)
    glob_feature_set_list = []
    # split based on models
    r2_all_models = []
    models_names = []
    sorted_r2 = {}
    for region, df_region in df_all_executions.groupby('model'):

        sorted_r2_per_model = []
        models_names.append(region)
        r2_list_per_featureset = []
        feature_set_list = []
        # print "df_region type: ", type(df_region)
        # print df_region.head(5)
        counter = 0
        for feature_set, df_subregion in df_region.groupby('features'):
            counter += 1
            if counter not in [1,2,3,4,5,6,20,21,22,23,24,25,26,27,28,29, 30]:
                continue
            df_subregion = df_subregion["r_2"]
            # print "r2_list: ",df_subregion.head(3)
            r2_list = [float(ele) for ele in df_subregion]
            ###### statistic deviation ############
            # st_dev = statistics.stdev(r2_list)
            # print st_dev
            #######################################3
            r2_mean = sum(r2_list)/len(r2_list)
            r2_list_per_featureset.append(r2_mean)
            # r2_list_per_featureset.append(st_dev)
            feature_set_list.append(str(feature_set))
            # print "dataset: {} \nmean r2: {} for model {}, feature {}\n".format(counter, r2_mean, region, feature_set)
            sorted_r2_per_model.append((r2_mean,feature_set)) # WANzny albo to albo  stdev
            # sorted_r2_per_model.append((st_dev, feature_set))
        sorted_r2_per_model.sort(key=lambda x: x[0])
        sorted_r2[region] = sorted_r2_per_model
        # print "region: {}, feature_set_list: {}".format(region, feature_set_list)
        # print "region {} ile featuresetow{}, ile r2 {}".format(region, len(feature_set_list),len(r2_list_per_featureset))
        x_pos = [i for i, _ in enumerate(feature_set_list)]

        glob_feature_set_list = feature_set_list[:]
        r2_all_models.append(r2_list_per_featureset)
        two_slots_posistion = [2*i for i in x_pos]
        plt.bar(two_slots_posistion, r2_list_per_featureset, color='green')
        plt.ylabel("R-kwadrat")
        plt.title(u"średnia(dla ponad 100 wykonań) R-kwadrat dla modelu {}".format(region))
        indexes = [ele +1  for ele in x_pos]
        plt.xticks(two_slots_posistion, indexes)
        plt.xlabel("wektory danych")
        test_list = [str(ele)for ele in range(len(feature_set_list))]

        for id , feature_set in enumerate(feature_set_list):
            print "{} : {}".format(id+1, feature_set)
        save = False
        if save:
            plt.savefig(LOG_DIR+"mean_r2_res_{}.png".format(region))
        else:
            # plt.show()
            pass
        plt.clf()
        # return 1
    x_pos = [i for i, _ in enumerate(glob_feature_set_list)]
    modes_len = len(r2_all_models)+2
    two_slots_posistion = [modes_len * i for i in x_pos]
    color_table = ["green", "blue", "red", "gray",  "orange", "yellow", "black"]
    possition_offset = -1
    all_bars_table = []
    # new_possitions = [ele + possition_offset for ele in two_slots_posistion]
    # foo = plt.bar(two_slots_posistion, r2_all_models[0], color='blue')
    for id, r2_per_model in enumerate(r2_all_models):
        print id , " -> ", r2_per_model
        new_possitions = [ele + possition_offset for ele in two_slots_posistion]
        print "new pos", new_possitions
        foo = plt.bar(new_possitions, r2_per_model, color=color_table[id])
        all_bars_table.append(foo)
        possition_offset += 1

    #

    models_names = ['Regresja liniowa', u'Sieć neuronowa', 'Las losowy', 'SVR']
    # plt.legend(tuple(all_bars_table), tuple(models_names) , loc='lower right')
    plt.legend(tuple(all_bars_table), tuple(models_names))

    # plt.legend([ "linear_regression", "mean_test", "mean_train"])
    plt.ylabel("R-kwadrat")
    plt.title("podsumowanie modeli")
    plt.title(u"")

    x_pos = [ele +1 for ele in x_pos]
    nr_fr_labels = [1,2,3,4,5,6,20,21,22,23,24,25,26,27,28,29, 30]
    fr_labels = [18, 19, 17, 8, 7, 10, 13, 15, 9, 16, 12, 14, 11, 6]
    negative_r2 = [22, 24, 25, 27, 28, 29, 30]
    wplyw_alg = [12, 25]
    fr_labels.sort()
    plt.xticks(two_slots_posistion, nr_fr_labels)
    plt.xlabel("zestaw danych")
    plt.show()


    # for model, content in sorted_r2.items():
    #     print model
    #     for r2, feature_set in content:
    #         print r2, "-> ", feature_set

    # print sorted_r2


if __name__ == '__main__':
    main()