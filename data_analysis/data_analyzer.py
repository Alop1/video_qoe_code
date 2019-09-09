import csv
from matplotlib import pyplot as plt
from collections import Counter
import os, re
import pandas as pd
import models as models
import data_preparation
import charts_generator
from charts_generator import create_correlation_chart, create_feature_bar_chart, create_quantitative_to_mos_chart, create_nominal_value_to_mos_chart
import random



def chart_module_based_on_merged_DB(merged_DB):
    # # old version create_nominal_value_to_mos_chart
    # # create_feature_bar_chart(merged_DB, "fsp", save=False)
    # # create_feature_bar_chart(merged_DB, "color_encoding", save=False)
    #
    create_nominal_value_to_mos_chart(merged_DB, "color_encoding", save=False)
    create_nominal_value_to_mos_chart(merged_DB, type="resolution", save=False)
    create_nominal_value_to_mos_chart(merged_DB, type='fps', save=False)

    create_quantitative_to_mos_chart(merged_DB, feature="spatialactivity",save=False)
    create_quantitative_to_mos_chart(merged_DB, feature="pillarbox",save=False)
    create_quantitative_to_mos_chart(merged_DB, feature="blockloss", save=False)
    create_quantitative_to_mos_chart(merged_DB, feature="blur", save=False)
    create_quantitative_to_mos_chart(merged_DB, feature="temporalact", save=False)
    create_quantitative_to_mos_chart(merged_DB, feature="exposure", save=False)
    create_quantitative_to_mos_chart(merged_DB, feature="contrast", save=False)
    create_quantitative_to_mos_chart(merged_DB, feature="brightness", save=False)

    create_quantitative_to_mos_chart(merged_DB, feature="interlace", save=False)
    # create_quantitative_to_mos_chart(merged_DB, feature="noise", save=False) # does not work
    create_quantitative_to_mos_chart(merged_DB, feature="slice", save=False)
    create_quantitative_to_mos_chart(merged_DB, feature="fps", save=False)

    create_quantitative_to_mos_chart(merged_DB, "Aggregate_vmaf", save=False)
    create_quantitative_to_mos_chart(merged_DB, "PSNR", save=False)
    create_quantitative_to_mos_chart(merged_DB, "SSIM", save=False)
    create_quantitative_to_mos_chart(merged_DB, "MS_SSIM", save=False)
    create_quantitative_to_mos_chart(merged_DB, "blockiness", save=False)
    create_quantitative_to_mos_chart(merged_DB, "frame", save=False)



@data_preparation.duration_measurement
def execute_LR(training_dataset, testing_dataset, *features):
    models.linear_regresion_model(training_dataset, testing_dataset, "PSNR", "Aggregate_vmaf")
    models.linear_regresion_model(training_dataset, testing_dataset, "PSNR", "SSIM")
    models.linear_regresion_model(training_dataset, testing_dataset, "Aggregate_vmaf", "SSIM")
    models.linear_regresion_model(training_dataset, testing_dataset, "Aggregate_vmaf")
    models.linear_regresion_model(training_dataset, testing_dataset, "PSNR")
    models.linear_regresion_model(training_dataset, testing_dataset, "SSIM")
    models.linear_regresion_model(training_dataset, testing_dataset, "MS_SSIM")
    models.linear_regresion_model(training_dataset, testing_dataset, "blockiness")
    # models.linear_regresion_model(training_dataset, testing_dataset, "frame")
    # models.linear_regresion_model(training_dataset, testing_dataset, "spatialactivity")
    # models.linear_regresion_model(training_dataset, testing_dataset, "pillarbox")
    models.linear_regresion_model(training_dataset, testing_dataset, "blockloss")
    models.linear_regresion_model(training_dataset, testing_dataset, "blur")
    # models.linear_regresion_model(training_dataset, testing_dataset, "temporalact")
    # models.linear_regresion_model(training_dataset, testing_dataset, "blockout")
    models.linear_regresion_model(training_dataset, testing_dataset, "exposure")
    models.linear_regresion_model(training_dataset, testing_dataset, "contrast")
    models.linear_regresion_model(training_dataset, testing_dataset, "brightness")
    # models.linear_regresion_model(training_dataset, testing_dataset, "fps")

    # models.linear_regresion_model(training_dataset, testing_dataset, "1920x1080", "352x288", "3840x2160", "640x480", "704x576" )
    # models.linear_regresion_model(training_dataset, testing_dataset, "duration")
    # print features[0]
    for feature_set in features[0]:
        # print "feature set ", feature_set
        models.linear_regresion_model(training_dataset, testing_dataset, *feature_set)


@data_preparation.duration_measurement
def execute_SVR(training_dataset, testing_dataset, *features):
    models.support_vector_regresion_model(training_dataset, testing_dataset, "PSNR", "Aggregate_vmaf")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "PSNR", "SSIM")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "Aggregate_vmaf", "SSIM")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "Aggregate_vmaf")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "PSNR")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "SSIM")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "MS_SSIM")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "blockiness")
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "frame")
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "spatialactivity")
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "pillarbox")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "blockloss")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "blur")
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "temporalact")
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "blockout")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "exposure")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "contrast")
    models.support_vector_regresion_model(training_dataset, testing_dataset, "brightness")
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "fps")

    # models.support_vector_regresion_model(training_dataset, testing_dataset, "1920x1080", "352x288", "3840x2160", "640x480", "704x576" )
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "duration")

    for feature_set in features[0]:
        # print "feature set ", feature_set[0]
        models.support_vector_regresion_model(training_dataset, testing_dataset, *feature_set)

@data_preparation.duration_measurement
def execute_NN(training_dataset, testing_dataset, *features):
    models.nn_model(training_dataset, testing_dataset, "PSNR", "Aggregate_vmaf")
    models.nn_model(training_dataset, testing_dataset, "PSNR", "SSIM")
    models.nn_model(training_dataset, testing_dataset, "Aggregate_vmaf", "SSIM")
    models.nn_model(training_dataset, testing_dataset, "Aggregate_vmaf")
    models.nn_model(training_dataset, testing_dataset, "PSNR")
    models.nn_model(training_dataset, testing_dataset, "SSIM")
    models.nn_model(training_dataset, testing_dataset, "MS_SSIM")
    models.nn_model(training_dataset, testing_dataset, "blockiness")
    # models.nn_model(training_dataset, testing_dataset, "frame")
    # models.nn_model(training_dataset, testing_dataset, "spatialactivity")
    # models.nn_model(training_dataset, testing_dataset, "pillarbox")
    models.nn_model(training_dataset, testing_dataset, "blockloss")
    models.nn_model(training_dataset, testing_dataset, "blur")
    # models.nn_model(training_dataset, testing_dataset, "temporalact")
    # models.nn_model(training_dataset, testing_dataset, "blockout")
    models.nn_model(training_dataset, testing_dataset, "exposure")
    models.nn_model(training_dataset, testing_dataset, "contrast")
    models.nn_model(training_dataset, testing_dataset, "brightness")
    # models.nn_model(training_dataset, testing_dataset, "fps")

    # models.nn_model(training_dataset, testing_dataset, "1920x1080", "352x288", "3840x2160", "640x480", "704x576" )
    # models.nn_model(training_dataset, testing_dataset, "duration")
    for feature_set in features[0]:
        # print "feature set ", feature_set[0]
        models.nn_model(training_dataset, testing_dataset, *feature_set)


@data_preparation.duration_measurement
def execute_RF(training_dataset, testing_dataset, *features):
    # models.RF_model(training_dataset, testing_dataset, "PSNR", "Aggregate_vmaf")
    # models.RF_model(training_dataset, testing_dataset, "PSNR", "SSIM")
    # models.RF_model(training_dataset, testing_dataset, "Aggregate_vmaf", "SSIM")
    # models.RF_model(training_dataset, testing_dataset, "Aggregate_vmaf")
    # models.RF_model(training_dataset, testing_dataset, "PSNR")
    # models.RF_model(training_dataset, testing_dataset, "SSIM")
    # models.RF_model(training_dataset, testing_dataset, "MS_SSIM")
    # models.RF_model(training_dataset, testing_dataset, "blockiness")
    # # models.RF_model(training_dataset, testing_dataset, "frame")
    # # models.RF_model(training_dataset, testing_dataset, "spatialactivity")
    # # models.RF_model(training_dataset, testing_dataset, "pillarbox")
    # models.RF_model(training_dataset, testing_dataset, "blockloss")
    # models.RF_model(training_dataset, testing_dataset, "blur")
    # # models.RF_model(training_dataset, testing_dataset, "temporalact")
    # # models.RF_model(training_dataset, testing_dataset, "blockout")
    # models.RF_model(training_dataset, testing_dataset, "exposure")
    # models.RF_model(training_dataset, testing_dataset, "contrast")
    # models.RF_model(training_dataset, testing_dataset, "brightness")
    # # models.RF_model(training_dataset, testing_dataset, "fps")
    #
    # # models.RF_model(training_dataset, testing_dataset, "1920x1080", "352x288", "3840x2160", "640x480", "704x576" )
    # # models.RF_model(training_dataset, testing_dataset, "duration")
    for feature_set in features[0]:
          # print "feature set ", feature_set[0]
          models.RF_model(training_dataset, testing_dataset, *feature_set)


@data_preparation.duration_measurement
def execute_LR_k_fold(training_dataset, testing_dataset, *features):
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "PSNR", "Aggregate_vmaf")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "PSNR", "SSIM")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "Aggregate_vmaf", "SSIM")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "Aggregate_vmaf")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "PSNR")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "SSIM")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "MS_SSIM")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "blockiness")
    # models.linear_regresion_mod_k_foldel(training_dataset, testing_dataset, "frame")
    # models.linear_regresion_mod_k_foldel(training_dataset, testing_dataset, "spatialactivity")
    # models.linear_regresion_mod_k_foldel(training_dataset, testing_dataset, "pillarbox")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "blockloss")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "blur")
    # models.linear_regresion_mod_k_foldel(training_dataset, testing_dataset, "temporalact")
    # models.linear_regresion_mod_k_foldel(training_dataset, testing_dataset, "blockout")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "exposure")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "contrast")
    models.linear_regresion_model_k_fold(training_dataset, testing_dataset, "brightness")
    # models.linear_regresion_model(training_dataset, testing_dataset, "fps")

    # models.linear_regresion_model(training_dataset, testing_dataset, "1920x1080", "352x288", "3840x2160", "640x480", "704x576" )
    # models.linear_regresion_model(training_dataset, testing_dataset, "duration")
    # print features[0]
    for feature_set in features[0]:
        # print "feature set ", feature_set
        models.linear_regresion_model_k_fold(training_dataset, testing_dataset, *feature_set)


@data_preparation.duration_measurement
def execute_SVR_k_fold(training_dataset, testing_dataset, *features):
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "PSNR", "Aggregate_vmaf")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "PSNR", "SSIM")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "Aggregate_vmaf", "SSIM")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "Aggregate_vmaf")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "PSNR")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "SSIM")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "MS_SSIM")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "blockiness")
    # models.support_vector_regresion_mod_k_foldel(training_dataset, testing_dataset, "frame")
    # models.support_vector_regresion_mod_k_foldel(training_dataset, testing_dataset, "spatialactivity")
    # models.support_vector_regresion_mod_k_foldel(training_dataset, testing_dataset, "pillarbox")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "blockloss")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "blur")
    # models.support_vector_regresion_mod_k_foldel(training_dataset, testing_dataset, "temporalact")
    # models.support_vector_regresion_mod_k_foldel(training_dataset, testing_dataset, "blockout")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "exposure")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "contrast")
    models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, "brightness")
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "fps")

    # models.support_vector_regresion_model(training_dataset, testing_dataset, "1920x1080", "352x288", "3840x2160", "640x480", "704x576" )
    # models.support_vector_regresion_model(training_dataset, testing_dataset, "duration")

    for feature_set in features[0]:
        # print "feature set ", feature_set[0]
        models.support_vector_regresion_model_k_fold(training_dataset, testing_dataset, *feature_set)

@data_preparation.duration_measurement
def execute_NN_k_fold(training_dataset, testing_dataset, *features):
    models.nn_model_k_fold(training_dataset, testing_dataset, "PSNR", "Aggregate_vmaf")
    models.nn_model_k_fold(training_dataset, testing_dataset, "PSNR", "SSIM")
    models.nn_model_k_fold(training_dataset, testing_dataset, "Aggregate_vmaf", "SSIM")
    models.nn_model_k_fold(training_dataset, testing_dataset, "Aggregate_vmaf")
    models.nn_model_k_fold(training_dataset, testing_dataset, "PSNR")
    models.nn_model_k_fold(training_dataset, testing_dataset, "SSIM")
    models.nn_model_k_fold(training_dataset, testing_dataset, "MS_SSIM")
    models.nn_model_k_fold(training_dataset, testing_dataset, "blockiness")
    # models.nn_model(training_dataset, testing_dataset, "frame")
    # models.nn_model(training_dataset, testing_dataset, "spatialactivity")
    # models.nn_model(training_dataset, testing_dataset, "pillarbox")
    models.nn_model_k_fold(training_dataset, testing_dataset, "blockloss")
    models.nn_model_k_fold(training_dataset, testing_dataset, "blur")
    # models.nn_model(training_dataset, testing_dataset, "temporalact")
    # models.nn_model(training_dataset, testing_dataset, "blockout")
    models.nn_model_k_fold(training_dataset, testing_dataset, "exposure")
    models.nn_model_k_fold(training_dataset, testing_dataset, "contrast")
    models.nn_model_k_fold(training_dataset, testing_dataset, "brightness")
    # models.nn_model(training_dataset, testing_dataset, "fps")

    # models.nn_model(training_dataset, testing_dataset, "1920x1080", "352x288", "3840x2160", "640x480", "704x576" )
    # models.nn_model(training_dataset, testing_dataset, "duration")
    for feature_set in features[0]:
        # print "feature set ", feature_set[0]
        models.nn_model_k_fold(training_dataset, testing_dataset, *feature_set)


@data_preparation.duration_measurement
def execute_RF_k_fold(training_dataset, testing_dataset, *features):
    models.RF_model_k_fold(training_dataset, testing_dataset, "PSNR", "Aggregate_vmaf")
    models.RF_model_k_fold(training_dataset, testing_dataset, "PSNR", "SSIM")
    models.RF_model_k_fold(training_dataset, testing_dataset, "Aggregate_vmaf", "SSIM")
    models.RF_model_k_fold(training_dataset, testing_dataset, "Aggregate_vmaf")
    models.RF_model_k_fold(training_dataset, testing_dataset, "PSNR")
    models.RF_model_k_fold(training_dataset, testing_dataset, "SSIM")
    models.RF_model_k_fold(training_dataset, testing_dataset, "MS_SSIM")
    models.RF_model_k_fold(training_dataset, testing_dataset, "blockiness")
    # models.RF_mod_k_foldel(training_dataset, testing_dataset, "frame")
    # models.RF_mod_k_foldel(training_dataset, testing_dataset, "spatialactivity")
    # models.RF_mod_k_foldel(training_dataset, testing_dataset, "pillarbox")
    models.RF_model_k_fold(training_dataset, testing_dataset, "blockloss")
    models.RF_model_k_fold(training_dataset, testing_dataset, "blur")
    # models.RF_mod_k_foldel(training_dataset, testing_dataset, "temporalact")
    # models.RF_mod_k_foldel(training_dataset, testing_dataset, "blockout")
    models.RF_model_k_fold(training_dataset, testing_dataset, "exposure")
    models.RF_model_k_fold(training_dataset, testing_dataset, "contrast")
    models.RF_model_k_fold(training_dataset, testing_dataset, "brightness")
    # models.RF_model(training_dataset, testing_dataset, "fps")

    # models.RF_model(training_dataset, testing_dataset, "1920x1080", "352x288", "3840x2160", "640x480", "704x576" )
    # models.RF_model(training_dataset, testing_dataset, "duration")
    for feature_set in features[0]:
          # print "feature set ", feature_set[0]
          models.RF_model_k_fold(training_dataset, testing_dataset, *feature_set)



def main(sim_quantity=1):


    ##############################################################
    #           PREPARE DATA
    ##############################################################

    pd_DB1, pd_DB_cif, pd_DB_cif4, pd_india_agh, pd_netflix1, pd_netflix2 = data_preparation.read_all_csv()
    pd_DB1, pd_DB_cif, pd_DB_cif4, pd_india_agh, pd_netflix1, pd_netflix2 = data_preparation.clear_data_phase_one(pd_DB1, pd_DB_cif, pd_DB_cif4, pd_india_agh, pd_netflix1, pd_netflix2)

    # normalize z-score MOS
    pd_DB1["Subject_score"] = data_preparation.z_score(pd_DB1["Subject_score"])
    pd_DB_cif["Subject_score"] = data_preparation.z_score(pd_DB_cif["Subject_score"])
    pd_DB_cif4["Subject_score"] = data_preparation.z_score(pd_DB_cif4["Subject_score"])
    pd_india_agh["Subject_score"] = data_preparation.z_score(pd_india_agh["Subject_score"])

    # column_names = pd_DB_cif.columns.values
    # print column_names


    #prepere nominal data - resolution
    # print pd.get_dummies(pd_DB1["Resolution"])
    # print pd.get_dummies(pd_DB_cif["Resolution"])
    # print pd.get_dummies(pd_DB_cif4["Resolution"])
    # print pd.get_dummies(pd_netflix1["Resolution"])
    # print pd.get_dummies(pd_netflix2["Resolution"])


    pd_DB1 = pd.concat([pd_DB1, pd.get_dummies(pd_DB1["Resolution"])], axis=1)
    pd_DB_cif = pd.concat([pd_DB_cif, pd.get_dummies(pd_DB_cif["Resolution"])], axis=1)
    pd_DB_cif4 = pd.concat([pd_DB_cif4, pd.get_dummies(pd_DB_cif4["Resolution"])], axis=1)
    pd_netflix1 = pd.concat([pd_netflix1, pd.get_dummies(pd_netflix1["Resolution"])], axis=1)
    pd_netflix2 = pd.concat([pd_netflix2, pd.get_dummies(pd_netflix2["Resolution"])], axis=1)
    pd_india_agh = pd.concat([pd_india_agh, pd.get_dummies(pd_india_agh["Resolution"])], axis=1)

    pd_DB1.drop(pd_DB1.tail(5).index, inplace=True)
    pd_DB_cif.drop(pd_DB_cif.tail(5).index, inplace=True)
    pd_DB_cif4.drop(pd_DB_cif4.tail(5).index, inplace=True)
    pd_netflix1.drop(pd_netflix1.tail(5).index, inplace=True)
    pd_netflix2.drop(pd_netflix2.tail(5).index, inplace=True)
    pd_india_agh.drop(pd_india_agh.tail(5).index, inplace=True)


    # df.drop(df.index[0], inplace=True)
    # df.drop(df.index[0], inplace=True)
    # df.drop(df.index[0], inplace=True)
    # df.drop(df.index[0], inplace=True)
    # df.drop(df.index[0], inplace=True)

    # pack data to subdata base on ref video
    splited_pd_DB_cif = data_preparation.split_DB_to_video_groups(pd_DB_cif)
    splited_pd_DB_cif4 = data_preparation.split_DB_to_video_groups(pd_DB_cif4)
    splited_pd_india_agh = data_preparation.split_DB_to_video_groups(pd_india_agh)
    splited_pd_netflix1 = data_preparation.split_DB_to_video_groups(pd_netflix1)
    splited_pd_DB1 = data_preparation.split_DB_to_video_groups(pd_DB1)
    # # split netflix
    splited_netflix2 = data_preparation.split_netflix_to_video_groups(pd_netflix2)


    ###############################################################
    #     prepare DB for k - fold CV
    ###############################################################
    # sub_DB_list =[]
    # sub_DB_list.extend(splited_pd_DB_cif)
    # sub_DB_list.extend(splited_pd_DB_cif4)
    # sub_DB_list.extend(splited_pd_india_agh)
    # sub_DB_list.extend(splited_pd_netflix1)
    # sub_DB_list.extend(splited_netflix2)
    # sub_DB_list.extend(splited_pd_DB1)
    # for group in sub_DB_list:
    #     group.dropna(subset=["fps"], inplace=True)
    #     group.dropna(subset=["blockiness"], inplace=True)
    #     group.dropna(subset=["MS_SSIM"], inplace=True)
    # random.shuffle(sub_DB_list)

    #############################################################
    #  prepare data fro MCCCV
    ############################################################
    #split data for training and testing dataset
    pd_cif_training, pd_cif_testing = data_preparation.split_DB_for_train_test(splited_pd_DB_cif)
    pd_cif4_trainig, pd_cif4_testing = data_preparation.split_DB_for_train_test(splited_pd_DB_cif4)
    pd_india_agh_training, pd_india_agh_testing = data_preparation.split_DB_for_train_test(splited_pd_india_agh)
    pd_netflix1_training, pd_netflix1_testing = data_preparation.split_DB_for_train_test(splited_pd_netflix1)
    pd_DB1_training, pd_DB1_testing = data_preparation.split_DB_for_train_test(splited_pd_DB1)
    pd_netflix2_trainig, pd_netflix2_testing = data_preparation.split_DB_for_train_test(splited_netflix2)
    # test = join_DBes(pd_cif_testing)
    training_dataset = data_preparation.join_DBes(pd_cif_training,pd_cif4_trainig,
                               pd_india_agh_training,pd_netflix1_training,
                               pd_DB1_training, pd_netflix2_trainig)
    # testing_dataset = join_DBes(pd_cif_testing)
    testing_dataset = data_preparation.join_DBes(pd_cif_testing, pd_cif4_testing,
                                pd_india_agh_testing, pd_DB1_testing,
                                pd_netflix1_testing, pd_netflix2_testing)
    # clear data phase two
    training_dataset.dropna(subset=["fps"], inplace=True)
    testing_dataset.dropna(subset=["fps"], inplace=True)
    training_dataset.dropna(subset=["blockiness"], inplace=True)
    testing_dataset.dropna(subset=["blockiness"], inplace=True)
    training_dataset.dropna(subset=["MS_SSIM"], inplace=True)
    testing_dataset.dropna(subset=["MS_SSIM"], inplace=True)

    ############################################################
    # # merge training and testing dataset just for chart purpose
    # merged_DB = pd.concat([training_dataset, testing_dataset], ignore_index=True)
    # merged_DB.dropna(subset=["fps"], inplace=False)
    # #  print "train + test shape : {} ".format(merged_DB.shape)
    # #  chart_module_based_on_merged_DB(merged_DB)
    # create_correlation_chart(merged_DB)
    # # import sys
    # # sys.exit(-1)
    ###########################################################

    # final dataset for training and testing
    # training_dataset                      )
    # testing_dataset

    ##############################################################
    #           MODELING
    ##############################################################

    # #no resoltutions - already studied
    features = []
    #all
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "blockiness",
                         "spatialactivity", "pillarbox", "blockloss", "blur", "temporalact",
                         "blockout", "exposure", "contrast", "brightness",
                         "duration", "1920x1080", "352x288", "3840x2160", "640x480","704x576",  'one_res',])
    # all no one_res
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "blockiness",
                    "spatialactivity", "pillarbox", "blockloss", "blur", "temporalact",
                    "blockout", "exposure", "contrast", "brightness",
                     "duration", "1920x1080", "352x288", "3840x2160", "640x480","704x576"])


    # NR
    features.append(["blockiness", "spatialactivity", "pillarbox", "blockloss", "blur",
                         "temporalact", "blockout", "exposure", "contrast", "brightness",
                          'duration', "1920x1080", "352x288", "3840x2160", "640x480","704x576"])
    # with one_res
    features.append(["blockiness", "spatialactivity", "pillarbox", "blockloss", "blur",
                    "temporalact", "blockout", "exposure", "contrast", "brightness",
                     'duration', "1920x1080", "352x288", "3840x2160", "640x480","704x576", 'one_res', ])
    # with res , without duration
    features.append(["blockiness", "spatialactivity", "pillarbox", "blockloss", "blur",
                     "temporalact", "blockout", "exposure", "contrast", "brightness",
                      "1920x1080", "352x288", "3840x2160", "640x480", "704x576", 'one_res', ])




    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM",])
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", 'duration'])
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "duration",  'one_res', ])
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "duration", 'one_res', "1920x1080", "352x288", "3840x2160", "640x480", "704x576", ])


    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "brightness"])
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "brightness", "exposure"])
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "blur", "blockloss", ])


    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "brightness",  'one_res'])
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "brightness", "exposure",  'one_res'])
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "blur", "blockloss",  'one_res'])

    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "brightness", "blockiness",
                     "blockloss", "blur",
                     'one_res'])
    # napodstawie wykresow pojedynczych cech
    features.append(["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM", "brightness", "blockiness",
                     "blockloss", "blur"
                    ])

    empty = []
    #
    print "processing Linear Regression..."
    execute_LR(training_dataset, testing_dataset, features)
    # # k-fold
    # execute_LR_k_fold(sub_DB_list, empty, features)
    #
    print "processing support_vector_regresion..."
    execute_SVR(training_dataset, testing_dataset, features)
    # k-fold
    # execute_SVR_k_fold(sub_DB_list, empty, features)

    print "processing MLPRegressor..."
    execute_NN(training_dataset, testing_dataset, features)
    # k-fold
    # execute_NN_k_fold(sub_DB_list, empty, features)

    print "processing RandomForestRegressor..."
    execute_RF(training_dataset, testing_dataset, features)
    # k-fold
    # execute_RF_k_fold(sub_DB_list, empty, features)



if __name__ == '__main__':
    main()

