from sklearn import datasets, linear_model, svm, neural_network
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
import re, os
import pandas as pd
from data_preparation import scale_data
# from data_analyzer import report

save_results = False

class Report(object):

    def __init__(self):
        self.report_file = open("reports/report_new_featuresets.txt", "a+")
        path = "reports/models_summary_csv_new_featuresets.csv"
        self.summary_df = pd.DataFrame(columns=['model_name', 'model_parameters', 'features_list', 'r2', 'database_size', 'coeff', 'ranking'])
        # print self.summary_df
        if os.path.isfile(path):
            self.models_summary_csv = open(path, "a+")
        else:
            self.models_summary_csv = open(path, "w+")
            self.header = "model, model_parameter, features, r_2, database_size, coeffs, features_ranking\n"
            self.models_summary_csv.write(self.header)

    def log_to_report_with_coef(self, model, r2, database_size,coef, *features):
        msg_to_log = "model : {}\nused records(training, testing): {}\nfeatures : {}\ncoefficients: {}\nr2 : {}\n;\n".format(
                    model, database_size, features, coef, r2)
        self.report_file.write(msg_to_log)

    def log_to_report_without_coef(self, model, r2, database_size,*features):
        msg_to_log = "model : {}\nused records(training, testing): {}\nfeatures : {}\nnr2 : {}\n;\n".format(
            model, database_size, features, r2)
        self.report_file.write(msg_to_log)

    def add_to_models_summary(self, model, features_list, r2, database_size, coeff="", ranking=[]):
        pattern = r"^(.*)(\(.+(\n\s*.*)*\))"
        match_obj = re.match(pattern, str(model))
        model_name = match_obj.group(1)
        model_parameters = match_obj.group(2)
        temp_dict = {'model_name': model_name, 'model_parameters': model_parameters,
                     'features_list': features_list, 'r2': r2, 'database_size': database_size,
                     'coeff' : coeff, 'ranking': ranking}
        row_pd = pd.Series(temp_dict)
        # Pass a series in append() to append a row in dataframe
        self.summary_df = self.summary_df.append(row_pd, ignore_index=True)

        # Pass a series in append() to append a row in dataframe
        # modDfObj = dfObj.append(pd.Series(['Raju', 21, 'Bangalore', 'India'], index=dfObj.columns), ignore_index=True)
        # print "series_pd: ", row_pd

        model_parameters = model_parameters.replace("\n", '')\
            .replace(',','|')\
            .replace("         ", ' ')\
            .replace("       ",'')
        database_size = str(database_size).replace(',', "|")
        features_list = str(features_list).replace(',', '|')
        self.summary_df.append(row_pd, ignore_index=True)

        if str(coeff):
            coeff = str(coeff).replace('\n', '')
        csv_input ='{},{},{},{},{},{},{}\n'.format(model_name, model_parameters, features_list, r2, database_size, coeff, ranking)
        # print "csv single_input {}".format(csv_input)
        self.models_summary_csv.write(csv_input)

    def __del__(self):
        # print "dataFrame: ", self.summary_df
        self.report_file.close()

report = Report()


def create_single_chart(vmaf_values):
    video_indexes = xrange(len(vmaf_values))
    g = plt.plot(video_indexes,vmaf_values, color='green', linestyle='solid').figure()
    plt.title("vmaf values")
    plt.ylabel("vmaf value")
    # plt.show()
    pass

def linear_regresion_model(training_data, testing_data, *features):
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
    #
    # scatter_chart =  plt.scatter(x_test, y_test, color='black')
    # line_chart_model  = plt.plot(x_test, predictions, color='blue', linewidth=3)
    # line_chart_mean = plt.plot(x_test, [y_test.mean() for ele in y_test], color='red', linewidth=3)
    # line_chart_mean_test = plt.plot(x_test, [y_train.mean() for ele in y_test], color='orange', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.legend([ "linear_regression", "mean_test", "mean_train"])
    # plt.show()

    database_size = (training_data.shape[0], testing_data.shape[0])

    if save_results:
        report.add_to_models_summary(model, list(features), r2, database_size, lm.coef_, fit.ranking_)

        # report.log_to_report_with_coef(model, r2, database_size, lm.coef_,*features)

def support_vector_regresion_model(training_data, testing_data, *features):
    x_train = training_data[list(features)]
    y_train = training_data["Subject_score"]
    x_test = testing_data[list(features)]
    y_test = testing_data["Subject_score"]
    svr = svm.SVR()

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
    # print "features: {}".format(features)
    # print "r2: ",r2_score(y_test, predictions), "\n"


    database_size = (training_data.shape[0], testing_data.shape[0])
    if save_results:
        report.add_to_models_summary(model, list(features), r2, database_size)

    # report.log_to_report_without_coef(model, r2, database_size, *features)

def nn_model(training_data, testing_data, *features):
    x_train = training_data[list(features)]
    y_train = training_data["Subject_score"]
    x_test = testing_data[list(features)]
    y_test = testing_data["Subject_score"]
    # x_train = scale_data(x_train)
    # x_test = scale_data(x_test)
    nn = MLPRegressor(hidden_layer_sizes=(13,13,13,13,13,13), max_iter=10000, random_state=9) # max_iter=500
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
    x_train = training_data[list(features)]
    y_train = training_data["Subject_score"]
    x_test = testing_data[list(features)]
    y_test = testing_data["Subject_score"]
    # x_train = scale_data(x_train)
    # x_test = scale_data(x_test)
    rf = RandomForestRegressor(n_estimators=1000,max_depth=10, random_state=42)
    model = rf.fit(x_train, y_train)
    # print "feature_list: ",list(features)
    # print "model: ", model
    # print "coef 0 ", nn.coef_
    predictions = rf.predict(x_test)
    # print "prediction {}".format(predictions)
    r2 = r2_score(y_test, predictions)
    # print "r2: ",r2_score(y_test, predictions)
    database_size = (training_data.shape[0], testing_data.shape[0])

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