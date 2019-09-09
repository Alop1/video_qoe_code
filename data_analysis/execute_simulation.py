
import data_analyzer
import time
import pandas as pd
import re, os


class PATH(object):     # todo - impr do not work for now, type name explicit in Report class
    def __init__(self):
        self.root_path = "reports/check_sim_quantity/new_tries"
        self.filename = "quantity_"


class Report(object):

    def __init__(self, sim_quantity='1', iter_no=0):
        print "start Report"
        self.iter_no = iter_no
        self.report_file = open("reports/rm.txt", "a+")
        self.path = "reports/cross_val/_test_sim_{}.txt".format(sim_quantity)
        self.summary_df = pd.DataFrame(
            columns=['model_name', 'model_parameters', 'features_list', 'r2', 'database_size', 'coeff', 'ranking'])
        # print self.summary_df
        if os.path.isfile(self.path):
            self.models_summary_csv = open(self.path, "a+")
        else:
            self.models_summary_csv = open(self.path, "w+")
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


start = time.time()
# sim_quantities = [i*10 for i in range(10, 20)]
COUNTER = 0

class Params():
    def __init__(self):
        ############# hyperparamteres selection ###########################
        # self.RF_PARAMS = [(x,y) for x in [2,6,8,10] for y in [10,20,40, 60, 70, 100]]
        # self.NN_PARAMS = [(x,y) for x in [1,2,3,4] for y in [2,3,5,9,13,15]]
        # self.SVR_PARAMS = [(x,y) for x in [0.1, 0.3, 0.4, 0.5] for y in [0.7, 10, 20, 30, 40, 60]]
        ############### final parmas ##########################################
        self.RF_PARAMS = [(x, y) for x in [8] for y in [60]]
        self.NN_PARAMS = [(x, y) for x in [3] for y in [13]]
        self.SVR_PARAMS = [(x, y) for x in [0.1] for y in [10]]
        self.CURRENT_RF_PARAM = []
        self.CURRENT_SVM_PARAM = []
        self.CURRENT_NN_PARAM = []

PARAM = Params()



# if __name__ == '__main__':

for j in xrange(1):        #ile roznych pomieszanych egzekucji
    global report

    # if j < 19:
    #     continue

    # PARAM.CURRENT_RF_PARAM = PARAM.RF_PARAMS[j]
    # PARAM.CURRENT_SVM_PARAM = PARAM.SVR_PARAMS[j]
    # PARAM.CURRENT_NN_PARAM = PARAM.NN_PARAMS[j]

    # print PARAM.CURRENT_SVM_PARAM
    # print PARAM.CURRENT_RF_PARAM

    COUNTER += 1
    report = Report(sim_quantity=str(j), iter_no=j)

    ###########################################
    ## test MCCV iteration
    # j = j*2
    # if j == 0:
    #     j = 1
    # sim_quantity = j
    ############################################

    sim_quantity = 70

    for i in xrange(sim_quantity):
        print "\niterarion no ", i
        # try:
        data_analyzer.main()
        # except:pass
    end = time.time()
    print "sim duration: {}".format(end - start)

