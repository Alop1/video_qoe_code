# -*- coding: utf-8 -*-
from sklearn import datasets, linear_model, svm, neural_network
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
# import pydot
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
import re, os



save_results = False






def create_single_chart(vmaf_values):
    from execute_simulation import report
    video_indexes = xrange(len(vmaf_values))
    g = plt.plot(video_indexes,vmaf_values, color='green', linestyle='solid').figure()
    plt.title("vmaf values")
    plt.ylabel("vmaf value")
    # plt.show()
    pass

def linear_regresion_model(training_data, testing_data, *features):
    # print features
    from execute_simulation import report
    x_train = training_data[list(features)]
    y_train = training_data["Subject_score"]
    x_test = testing_data[list(features)]
    y_test = testing_data["Subject_score"]
    lm = linear_model.LinearRegression()
    ref = RFE(lm, 3)
    fit = ref.fit(x_train, y_train)
    model = lm.fit(x_train, y_train)
    # print "fearures: ", list(features)
    pattern = r"^(.*)(\(.+(\n\s*.*)*\))"
    match_obj = re.match(pattern, str(model))
    model_name = match_obj.group(1)
    model_parameters = match_obj.group(2)
    # print "model: ", str(model)
    # print "coef 0 ", lm.coef_
    predictions = lm.predict(x_test)
    # print("Num Features: %d") % fit.n_features_
    # print("Selected Features: %s") % fit.support_
    # print("Feature Ranking: %s") % fit.ranking_
    # print "prediction {}".format(predictions)
    r2 = r2_score(y_test, predictions)
    # print "lineraREgression r2: ",r2_score(y_test, predictions)
    r2_model_part = ((predictions - y_test ) ** 2).sum()
    r2_mean_part = ((y_test - y_test.mean()) ** 2).sum()
    # print "r2 mean part : ", r2_mean_part
    # print "r2 model part : ", r2_model_part
    my_r2 =  (1 - r2_model_part/r2_mean_part)
    # print "my r2 : ", my_r2
    # Plot outputs

    # print "len test_X {}, test x {} \n\n".format(len(x_test),x_test)
    # print "len test_y {}, test y {} \n\n".format(len(y_test),y_test)

    # print_negative_r2(features, predictions, x_test, y_test, y_train)

    database_size = (training_data.shape[0], testing_data.shape[0])

    if save_results:
        report.add_to_models_summary(model, list(features), r2, database_size, lm.coef_, fit.ranking_)

        # report.log_to_report_with_coef(model, r2, database_size, lm.coef_,*features)
def map_to_polish(): # todo impr - convert to class
    map_dictionary = {
        "blockiness": u"blokowość",
        "spatialactivity": u"aktywność przest.",
        "pillarbox": u"pillarbox",
        "blockloss": u'straty bloków',
        "blur": u'rozmycie',
        "temporalact": u'ekspozycja',
        "contrast": u"kontrast",
        "brightness":u"jasność",
        "duration": u"czas trwania",
        "Aggregate_vmaf": "VMAF",
        "MS_SSIM": "MS-SSIM",
        "PSNR":"PSNR",
        "SSIM":"SSIM",
        "one_res": u"ilość rozdzielczości"
        }
    return map_dictionary

def print_negative_r2(features,predictions, x_test, y_test, y_train):
    from execute_simulation import report
    mapping_dictionary = map_to_polish()
    try:
        title = mapping_dictionary[(features[0])]
        scatter_chart = plt.scatter(x_test, y_test, color='black')
        line_chart_model = plt.plot(x_test, predictions, color='blue', linewidth=3)
        line_chart_mean = plt.plot(x_test, [y_test.mean() for ele in y_test], color='red', linewidth=3)
        line_chart_mean_test = plt.plot(x_test, [y_train.mean() for ele in y_test], color='orange', linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
        plt.legend(["regresja liniowa", u"średnia dla danych testowych", u"średnia dla danych treningowych"])
        plt.show()
    except (ValueError, KeyError):
        pass

def support_vector_regresion_model(training_data, testing_data, *features):
    from execute_simulation import report, PARAM
    x_train = training_data[list(features)]
    y_train = training_data["Subject_score"]
    x_test = testing_data[list(features)]
    y_test = testing_data["Subject_score"]


    CURR_ITER = report.iter_no

    svr = svm.SVR(kernel="rbf", C=PARAM.SVR_PARAMS[CURR_ITER][1], epsilon=PARAM.SVR_PARAMS[CURR_ITER][0]) #"linear"

    rfe = RFE(svr, 25)
    fit = rfe.fit(x_train, y_train )
    # print("Num Features: %d") % fit.n_features_
    # print("Selected Features: %s") % fit.support_
    # print("Feature Ranking: %s") % fit.ranking_

    model = svr.fit(x_train, y_train)
    # print "fearures: ", list(features)
    # print "model: ", model
    # print "coef 0 ", svr.coef_
    predictions = svr.predict(x_test)
    # print "prediction {}".format(predictions)
    r2 = r2_score(y_test, predictions)
    # print "features: {}".format(features)
    # print "r2: ",r2_score(y_test, predictions), "\n"


    database_size = (training_data.shape[0], testing_data.shape[0])
    if save_results:
        report.add_to_models_summary(model, list(features), r2, database_size)

    # report.log_to_report_without_coef(model, r2, database_size, *features)

def nn_model(training_data, testing_data, *features):
    from execute_simulation import report, PARAM
    CURR_ITER = report.iter_no
    # print  "CURR iter ", CURR_ITER
    x_train = training_data[list(features)]
    y_train = training_data["Subject_score"]
    x_test = testing_data[list(features)]
    y_test = testing_data["Subject_score"]
    # x_train = scale_data(x_train)
    # x_test = scale_data(x_test)
    hidden_l = tuple(( PARAM.NN_PARAMS[CURR_ITER][1] for ele in xrange(PARAM.NN_PARAMS[CURR_ITER][0])))
    nn = MLPRegressor(hidden_layer_sizes=hidden_l, max_iter=5000) # max_iter=500 random_state=9
    model = nn.fit(x_train, y_train)
    # print "fearures: ", list(features)
    # print "model: ", model
    # print "coef 0 ", nn.coef_
    predictions = nn.predict(x_test)
    # print "prediction {}".format(predictions)
    r2 = r2_score(y_test, predictions)
    # print "r2: ",r2_score(y_test, predictions)
    database_size = (training_data.shape[0], testing_data.shape[0])


    if save_results:
        report.add_to_models_summary(model, list(features), r2, database_size)
    # report.log_to_report_with_coef(model, r2, database_size, lm.coef_,*features)

def RF_model(training_data, testing_data, *features):
    from execute_simulation import report, PARAM
    mapping_dir = map_to_polish()
    CURR_ITER = report.iter_no
    x_train = training_data[list(features)]
    y_train = training_data["Subject_score"]
    x_test = testing_data[list(features)]
    y_test = testing_data["Subject_score"]
    # x_train = scale_data(x_train)
    # x_test = scale_data(x_test)
    rf = RandomForestRegressor(n_estimators=PARAM.RF_PARAMS[CURR_ITER][1], max_depth=PARAM.RF_PARAMS[CURR_ITER][0]) #random_state=42
    model = rf.fit(x_train, y_train)
    # print "feature_list: ",list(features)
    # print "model: ", model
    # print "coef 0 ", nn.coef_
    predictions = rf.predict(x_test)
    # print "prediction {}".format(predictions)
    r2 = r2_score(y_test, predictions)
    # print "r2: ",r2_score(y_test, predictions)
    database_size = (training_data.shape[0], testing_data.shape[0])


    # Sort the feature importances by most important first
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                           zip(list(features), importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    feature_importances_pl = [(mapping_dir[feature[0]], feature[1]) for feature in feature_importances]
    featues_score = [feature[1] for feature in feature_importances]
    plt.barh(range(len(feature_importances_pl)), featues_score, align='center', alpha=0.5)
    plt.yticks(range(len(feature_importances_pl)), [feature[0] for feature in feature_importances_pl])
    plt.xlabel(u'Istotność cech')
    if save_results:
        report.add_to_models_summary(model, list(features), r2, database_size)




def print_tree(rf, feature_list):
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file='./reports/tree_deep3.dot', feature_names=feature_list, rounded=True, precision=1)
    # Use dot file to create a graph
    (graph,) = pydot.graph_from_dot_file('./reports/tree_deep3.dot')
    # Write graph to a png file
    graph.write_png('./reports/tree_deep3.png')
    from graphviz import Source
    path = './reports/tree_deep3.dot'
    s = Source.from_file(path)
    s.view()


def linear_regresion_model_k_fold(training_data, testing_data, *features):
    # print features
    from execute_simulation import report
    from data_preparation import join_DBes

    start_id = 0
    end_id = 22
    r_2_corss_list = []
    model = ""
    for crossVal_idx in xrange(5):
        tmp_training = list(training_data)
        tmp_testing = training_data[start_id:end_id]
        del tmp_training[start_id:end_id]

        ready_traing_data = join_DBes(tmp_training)
        ready_testing_data = join_DBes(tmp_testing)

        x_train = ready_traing_data[list(features)]
        y_train = ready_traing_data["Subject_score"]
        x_test = ready_testing_data[list(features)]
        y_test = ready_testing_data["Subject_score"]
        lm = linear_model.LinearRegression()
        ref = RFE(lm, 3)
        fit = ref.fit(x_train, y_train)
        model = lm.fit(x_train, y_train)
        # print "fearures: ", list(features)
        pattern = r"^(.*)(\(.+(\n\s*.*)*\))"
        match_obj = re.match(pattern, str(model))
        model_name = match_obj.group(1)
        model_parameters = match_obj.group(2)
        # print "model: ", str(model)
        # print "coef 0 ", lm.coef_
        predictions = lm.predict(x_test)
        # print("Num Features: %d") % fit.n_features_
        # print("Selected Features: %s") % fit.support_
        # print("Feature Ranking: %s") % fit.ranking_
        # print "prediction {}".format(predictions)
        r2 = r2_score(y_test, predictions)
        r_2_corss_list.append(r2)
        # print "lineraREgression r2: ",r2_score(y_test, predictions)
        r2_model_part = ((predictions - y_test) ** 2).sum()
        r2_mean_part = ((y_test - y_test.mean()) ** 2).sum()
        # print "r2 mean part : ", r2_mean_part
        # print "r2 model part : ", r2_model_part
        my_r2 = (1 - r2_model_part / r2_mean_part)
        # print "my r2 : ", my_r2
        # Plot outputs

        # print "len test_X {}, test x {} \n\n".format(len(x_test),x_test)
        # print "len test_y {}, test y {} \n\n".format(len(y_test),y_test)

        # print_negative_r2(features, predictions, x_test, y_test, y_train)

    final_r2 = sum(r_2_corss_list) / len(r_2_corss_list)
    # print features
    # print "r2: ", final_r2
    # print "\n"

    database_size = (ready_traing_data.shape[0], ready_testing_data.shape[0])
    if save_results:
        report.add_to_models_summary(model, list(features), final_r2, database_size)


def support_vector_regresion_model_k_fold(training_data, testing_data, *features):
    from execute_simulation import report, PARAM
    from data_preparation import join_DBes

    CURR_ITER = report.iter_no

    start_id = 0
    end_id = 22
    r_2_corss_list = []
    model = ""
    for crossVal_idx in xrange(5):
        tmp_training = list(training_data)
        tmp_testing = training_data[start_id:end_id]
        del tmp_training[start_id:end_id]

        ready_traing_data = join_DBes(tmp_training)
        ready_testing_data = join_DBes(tmp_testing)

        x_train = ready_traing_data[list(features)]
        y_train = ready_traing_data["Subject_score"]
        x_test = ready_testing_data[list(features)]
        y_test = ready_testing_data["Subject_score"]

        # svr = svm.SVR(kernel="rbf", C=PARAM.SVR_PARAMS[CURR_ITER][1], epsilon=PARAM.SVR_PARAMS[CURR_ITER][0]) #"linear"
        svr = svm.SVR(kernel="rbf", C=10, epsilon=0.1)  # "linear"

        rfe = RFE(svr, 25)
        fit = rfe.fit(x_train, y_train)
        # print("Num Features: %d") % fit.n_features_
        # print("Selected Features: %s") % fit.support_
        # print("Feature Ranking: %s") % fit.ranking_

        model = svr.fit(x_train, y_train)
        # print "fearures: ", list(features)
        # print "model: ", model
        # print "coef 0 ", svr.coef_
        predictions = svr.predict(x_test)
        # print "prediction {}".format(predictions)
        r2 = r2_score(y_test, predictions)
        r_2_corss_list.append(r2)
    # print "features: {}".format(features)
    # print "r2: ",r2_score(y_test, predictions), "\n"

    final_r2 = sum(r_2_corss_list) / len(r_2_corss_list)
    # print features
    # print "r2: ", final_r2
    # print "\n"

    database_size = (ready_traing_data.shape[0], ready_testing_data.shape[0])
    if save_results:
        report.add_to_models_summary(model, list(features), final_r2, database_size)

    # report.log_to_report_without_coef(model, r2, database_size, *features)


def nn_model_k_fold(training_data, testing_data, *features):
    from execute_simulation import report, PARAM
    from data_preparation import join_DBes
    CURR_ITER = report.iter_no
    # print  "CURR iter ", CURR_ITER
    start_id = 0
    end_id = 22
    r_2_corss_list = []
    model = ""
    for crossVal_idx in xrange(5):
        tmp_training = list(training_data)
        tmp_testing = training_data[start_id:end_id]
        del tmp_training[start_id:end_id]

        ready_traing_data = join_DBes(tmp_training)
        ready_testing_data = join_DBes(tmp_testing)

        x_train = ready_traing_data[list(features)]
        y_train = ready_traing_data["Subject_score"]
        x_test = ready_testing_data[list(features)]
        y_test = ready_testing_data["Subject_score"]
        # x_train = scale_data(x_train)
        # x_test = scale_data(x_test)
        # hidden_l = tuple(( PARAM.NN_PARAMS[CURR_ITER][1] for ele in xrange(PARAM.NN_PARAMS[CURR_ITER][0])))
        hidden_l = (3, 13)
        nn = MLPRegressor(hidden_layer_sizes=hidden_l, max_iter=5000)  # max_iter=500 random_state=9
        model = nn.fit(x_train, y_train)
        # print "fearures: ", list(features)
        # print "model: ", model
        # print "coef 0 ", nn.coef_
        predictions = nn.predict(x_test)
        # print "prediction {}".format(predictions)
        r2 = r2_score(y_test, predictions)
        r_2_corss_list.append(r2)
    # print "r2: ",r2_score(y_test, predictions)
    database_size = (ready_traing_data.shape[0], ready_testing_data.shape[0])

    final_r2 = sum(r_2_corss_list) / len(r_2_corss_list)
    # print features
    # print "r2: ",final_r2
    # print "\n"

    if save_results:
        report.add_to_models_summary(model, list(features), final_r2, database_size)


def RF_model(training_data, testing_data, *features):
    from execute_simulation import report, PARAM
    from data_preparation import join_DBes
    # CURR_ITER = report.iter_no

    print "dlugosc danych ", len(training_data)
    mapping_dir = map_to_polish()
    # for x in range(len(training_data)):
    start_id = 0
    end_id = 22
    r_2_corss_list = []
    model = ""
    for crossVal_idx in xrange(5):
        tmp_training = list(training_data)
        tmp_testing = training_data[start_id:end_id]
        del tmp_training[start_id:end_id]
        # print "full data size ", len(training_data)
        # print "start id ", start_id
        # print "end id ", end_id
        # print "test lengh ", len(tmp_testing)
        # print "train lenght ", len(tmp_training)

        start_id = end_id
        end_id += 20

        ready_traing_data = join_DBes(tmp_training)
        ready_testing_data = join_DBes(tmp_testing)
        # if list(features) != ["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "duration",  'one_res', ]:
        #     return 0
        # print  list(features)

        # todo przygotowac odpowiedni traing and testing data, z zbioru podgrup

        x_train = ready_traing_data[list(features)]
        y_train = ready_traing_data["Subject_score"]
        x_test = ready_testing_data[list(features)]
        y_test = ready_testing_data["Subject_score"]
        # x_train = scale_data(x_train)
        # x_test = scale_data(x_test)
        import cPickle as pickle
        # rf = RandomForestRegressor(n_estimators=PARAM.RF_PARAMS[CURR_ITER][1], max_depth=PARAM.RF_PARAMS[CURR_ITER][0]) #random_state=42
        rf = RandomForestRegressor(n_estimators=60, max_depth=8)  # random_state=42

        model = rf.fit(x_train, y_train)
        # print "feature_list: ",list(features)
        # print "model: ", model
        # print "coef 0 ", nn.coef_
        predictions = rf.predict(x_test)
        # print "prediction {}".format(predictions)
        r2 = r2_score(y_test, predictions)
        r_2_corss_list.append(float(r2))
        importances = list(rf.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(list(features), importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        feature_importances_pl = [(mapping_dir[feature[0]], feature[1]) for feature in feature_importances]
        featues_score = [feature[1] for feature in feature_importances]

        plt.barh(range(len(feature_importances_pl)), featues_score, align='center', alpha=0.5)
        plt.yticks(range(len(feature_importances_pl)), [feature[0] for feature in feature_importances_pl])
        plt.xlabel(u'Istotność cech')
        # plt.title('Programming language usage')

        plt.show()

        # Print out the feature and importances
        # print "istotność: ", feature_importances
        # for pair in feature_importances_pl:
        #     print 'Variable: {:20} Importance: {}'.format(*pair)
        # print "\n\n"
        # break

    final_r2 = sum(r_2_corss_list) / len(r_2_corss_list)
    # print features
    # print "r2: ",final_r2
    # print "\n"

    # print 'serializacja'
    # pickle.dump(rf, open('las_losowy_model.pkl', 'w' ))

    database_size = (ready_traing_data.shape[0], ready_testing_data.shape[0])

    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                           zip(list(features), importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    # print "feature_importances: ", feature_importances
    # for pair in feature_importances:
    #     print 'Variable: {:20} Importance: {}'.format(*pair)
    # Save the tree as a png image
    # print_tree(rf, list(features))
    # report.log_to_report_with_coef(model, r2, database_size, lm.coef_,*features)
    if save_results:
        report.add_to_models_summary(model, list(features), final_r2, database_size)