import pandas as pd
from matplotlib import pyplot as plt


def main():
    # read CSV
    df_all_executions = pd.read_csv(r"reports/models_summary_csv_new_featuresets_clean.csv", skipinitialspace=True)
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
            df_subregion = df_subregion["r_2"]
            # print "r2_list: ",df_subregion.head(3)
            r2_list = [float(ele) for ele in df_subregion]
            r2_mean = sum(r2_list)/len(r2_list)
            r2_list_per_featureset.append(r2_mean)
            feature_set_list.append(str(feature_set))
            # print "dataset: {} \nmean r2: {} for model {}, feature {}\n".format(counter, r2_mean, region, feature_set)
            sorted_r2_per_model.append((r2_mean,feature_set))
        sorted_r2_per_model.sort(key=lambda x: x[0])
        sorted_r2[region] = sorted_r2_per_model
        # print "region: {}, feature_set_list: {}".format(region, feature_set_list)
        # print "region {} ile featuresetow{}, ile r2 {}".format(region, len(feature_set_list),len(r2_list_per_featureset))
        x_pos = [i for i, _ in enumerate(feature_set_list)]

        glob_feature_set_list = feature_set_list[:]
        r2_all_models.append(r2_list_per_featureset)
        two_slots_posistion = [2*i for i in x_pos]
        plt.bar(two_slots_posistion, r2_list_per_featureset, color='green')
        plt.ylabel("mean r2")
        plt.title("mean(for 150 executions) r2 in model {}".format(region))
        plt.xticks(two_slots_posistion, two_slots_posistion)
        plt.xlabel("featureset")
        test_list = [str(ele)for ele in range(len(feature_set_list))]

        # for id , feature_set in enumerate(feature_set_list):
        #     print "{} : {}".format(id, feature_set)
        save = False
        if save:
            plt.savefig("reports/r2_charts/mean_r2_res_{}.png".format(region))
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
        new_possitions = [ele + possition_offset for ele in two_slots_posistion]
        foo = plt.bar(new_possitions, r2_per_model, color=color_table[id])
        all_bars_table.append(foo)
        possition_offset += 1
    #
    plt.legend(tuple(all_bars_table), tuple(models_names))
    # plt.legend([ "linear_regression", "mean_test", "mean_train"])
    plt.ylabel("mean r2")
    plt.title("all models summary")
    plt.xticks(two_slots_posistion, x_pos)
    plt.xlabel("featureset")
    plt.show()

    for model, content in sorted_r2.items():
        print model
        for r2, feature_set in content:
            print r2, "-> ", feature_set

    # print sorted_r2

if __name__ == '__main__':
    main()