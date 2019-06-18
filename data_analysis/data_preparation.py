from matplotlib import pyplot as plt
import pandas as pd
import random, time


def create_single_chart(vmaf_values):
    video_indexes = xrange(len(vmaf_values))
    g = plt.plot(video_indexes,vmaf_values, color='green', linestyle='solid').figure()
    plt.title("vmaf values")
    plt.ylabel("vmaf value")
    # plt.show()
    pass

def read_all_csv():
    pd.set_option("display.max_colwidth", 100)

    pd_DB1 = pd.read_csv(r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\DB1.csv", skipinitialspace=True)
    pd_DB_cif = pd.read_csv(r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\DB_cif.csv", skipinitialspace=True)
    pd_DB_cif4 = pd.read_csv(r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\DB_cif4.csv", skipinitialspace=True)
    pd_india_agh = pd.read_csv(r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\india_agh.csv", skipinitialspace=True)
    pd_netflix1 = pd.read_csv(r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\netflix1.csv", skipinitialspace=True)
    pd_netflix2 = pd.read_csv(r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\netflix2.csv", skipinitialspace=True)

    return pd_DB1, pd_DB_cif, pd_DB_cif4, pd_india_agh, pd_netflix1, pd_netflix2

def clear_data_phase_one(pd_DB1, pd_DB_cif, pd_DB_cif4, pd_india_agh, pd_netflix1, pd_netflix2):
    # rm none value from DB
    pd_DB_cif = pd_DB_cif[pd_DB_cif.Subject_score != "none"]
    pd_DB_cif4 = pd_DB_cif4[pd_DB_cif4.Subject_score != "none"]
    pd_india_agh = pd_india_agh[pd_india_agh.Subject_score != "none"]
    pd_DB1 = pd_DB1[pd_DB1.Subject_score != "none"]
    pd_netflix1 = pd_netflix1[pd_netflix1.Subject_score != "none"]
    # rm NaN PSNR value
    pd_DB_cif = pd_DB_cif[pd_DB_cif.PSNR.notna()]
    pd_DB_cif4 = pd_DB_cif4[pd_DB_cif4.PSNR.notna()]
    pd_DB1 = pd_DB1[pd_DB1.PSNR.notna()]
    pd_india_agh = pd_india_agh[pd_india_agh.PSNR.notna()]
    pd_netflix1 = pd_netflix1[pd_netflix1.PSNR.notna()]
    pd_netflix2 = pd_netflix2[pd_netflix2.PSNR.notna()]
    # rm NaN SSIM value
    pd_DB_cif = pd_DB_cif[pd_DB_cif.SSIM.notna()]
    pd_DB_cif4 = pd_DB_cif4[pd_DB_cif4.SSIM.notna()]
    pd_DB1 = pd_DB1[pd_DB1.SSIM.notna()]
    pd_india_agh = pd_india_agh[pd_india_agh.SSIM.notna()]
    pd_netflix1 = pd_netflix1[pd_netflix1.SSIM.notna()]
    pd_netflix2 = pd_netflix2[pd_netflix2.SSIM.notna()]
    # rm #DIV/0!
    pd_DB_cif = pd_DB_cif[pd_DB_cif.blockiness.notna()]
    pd_DB_cif4 = pd_DB_cif4[pd_DB_cif4.blockiness.notna()]
    pd_DB1 = pd_DB1[pd_DB1.blockiness.notna()]
    pd_india_agh = pd_india_agh[pd_india_agh.blockiness.notna()]
    pd_netflix1 = pd_netflix1[pd_netflix1.blockiness.notna()]
    # netflix 2 later
    pd_DB_cif["Subject_score"] = [float(ele) for ele in pd_DB_cif["Subject_score"]]
    pd_DB_cif4["Subject_score"] = [float(ele) for ele in pd_DB_cif4["Subject_score"]]
    pd_india_agh["Subject_score"] = [float(ele) for ele in pd_india_agh["Subject_score"]]
    pd_DB1["Subject_score"] = [float(ele) for ele in pd_DB1["Subject_score"]]
    pd_netflix1["Subject_score"] = [float(ele) for ele in pd_netflix1["Subject_score"]]
    pd_netflix2["Subject_score"] = [float(ele) for ele in pd_netflix2["Subject_score"]]

    return pd_DB1, pd_DB_cif, pd_DB_cif4, pd_india_agh, pd_netflix1, pd_netflix2


def z_score(column):
    new_coulum = (column - column.mean())/ column.std()
    return new_coulum

def split_DB_to_video_groups(DB):
    # column_names = DB.columns.values
    # print column_names
    dict_DB_subdatabases = []
    current_group_name = 'foreman_'
    current_group_name = DB["Ref_video"][0]
    start_idx = 0
    end_idx = 0
    add_flag = False
    for idx in xrange(DB.shape[0]):
        if current_group_name != DB.iloc[idx]["Ref_video"]:
            end_idx = idx
            add_flag = True
        if idx + 1 == DB.shape[0]:
            end_idx = idx +1
            add_flag = True
        if add_flag:
            dict_DB_subdatabases.append(DB.iloc[start_idx:end_idx])
            current_group_name = DB.iloc[idx]["Ref_video"]
            start_idx = end_idx
            add_flag = False

    # print dict_DB_subdatabases[-1]
    return dict_DB_subdatabases

def split_netflix_to_video_groups(DB):
    # column_names = DB.columns.values
    # print column_names
    dict_DB_subdatabases = []
    current_group_name = 'foreman_'
    current_group_name = DB["Source_file"][0][-1]
    # print "current_group_name: ", current_group_name
    # sys.exit(-1)
    start_idx = 0
    end_idx = 0
    add_flag = False
    temp_list = DB["Source_file"]
    move_index_according_except = 0
    for idx, source_file in enumerate(DB.Source_file):
        # print "idx: ", idx
        try:
            # print "DB['Source_file': ",source_file, "file: ", idx

            # print "current_group_name {} DB[Source_file][0][-1]: {}".format(current_group_name, DB["Source_file"][idx][-1])
            if current_group_name >= DB["Source_file"][idx][-1]:
                end_idx = idx
                add_flag = True
            if idx + 1 == DB.shape[0]:
                end_idx = idx + 1
                add_flag = True
            if add_flag:
                # print DB.iloc[start_idx:end_idx - move_index_according_except]
                dict_DB_subdatabases.append(DB.iloc[start_idx:end_idx - move_index_according_except])
                # print ""
                start_idx = end_idx - move_index_according_except
                add_flag = False
                move_index_according_except = 0
            current_group_name = DB["Source_file"][idx][-1]
        except Exception as e :
            # print "except :", e
            move_index_according_except += 1

    # print dict_DB_subdatabases[0]["Source_file"]
    # print "***"
    # print dict_DB_subdatabases[-2]["Source_file"]
    # # print dict_DB_subdatabases[11]["Source_file"]
    # # print dict_DB_subdatabases[14]["Source_file"]
    # print "!!"
    # print dict_DB_subdatabases[-1]["Source_file"]

    return dict_DB_subdatabases


def split_DB_for_train_test(DB, isRandom=True):
    # print Counter(foo)
    # bar = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
    # print len(DB)
    # print (len(DB) * 20) / 100
    test_data_quantity = (len(DB) * 20) / 100
    if isRandom:
        test_idxes = random.sample(xrange(len(DB)-1), test_data_quantity)
        test_idxes.sort()
        test_idxes = test_idxes[::-1]
    else:
        test_idxes = range(test_data_quantity)
        for id, el in enumerate(test_idxes):
            if el%2: test_idxes[id] = - el

    testing_data = []
    map(lambda x: testing_data.append(DB[x]), test_idxes)
    training_data = DB[:]
    map(lambda x: training_data.pop(x), test_idxes)
    # print len(training_data)
    return training_data, testing_data

def join_DBes(*args):
    all_pd_frames = []
    for arg in args:
        all_pd_frames.extend(arg)
    # print "ilosc podbaz: ",len(all_pd_frames)
    merged_DB = pd.concat(all_pd_frames, ignore_index=True)
    # print "shape after merge: ", merged_DB.shape
    return merged_DB


def scale_data(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
     # Don't cheat - fit only on training data
    scaler.fit(data)
    X_train = scaler.transform(data)
     # apply same transformation to test data
    scaled_data = scaler.transform(data)
    return scaled_data


def duration_measurement(func):
        def wraper(*c):
            start = time.time()
            func(*c)
            end = time.time()
            print "durations: {}".format(end - start)
        return wraper

