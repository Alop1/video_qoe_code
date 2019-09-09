import csv
from matplotlib import pyplot as plt
from collections import Counter
from abc import ABCMeta, abstractmethod

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

class DataObj(object):
    def __init__(self):
        self.attribute_list = []
        self.name = []
        self.vmaf = []
        self.psnr = []
        self.ssim = []
        self.ms_ssim = []
        self.resolution = []
        self.ref_video = []
        self.color_encoding = []
        self.mos = []
        self.STRRED = []

        self.frame = []
        self.blockiness = []
        self.spatialactivity = []
        self.letterbox = []
        self.pillarbox = []
        self.blockloss = []
        self.blur = []
        self.temporalact = []
        self.blockout = []
        self.freezing = []
        self.exposure = []
        self.contrast = []
        self.brightness = []
        self.interlace = []
        self.noise = []
        self.slice = []
        self.flickering = []
        self.fps = []

    def load_data(self, path):
        with open(path, 'rb') as f :
            reader = csv.DictReader(f, delimiter=',')
            self.attribute_list = reader.fieldnames
            for row in reader:
                row_without_spaces = {key.replace(" ", ''): value.replace(" ", '') for key, value in row.items()}
                self.name.append(row_without_spaces['Source_file'])
                self.vmaf.append(row_without_spaces['Aggregate_vmaf'])
                self.psnr.append(row_without_spaces['PSNR'])
                self.ssim.append(row_without_spaces['SSIM'])
                self.ms_ssim.append(row_without_spaces['MS_SSIM'])
                self.resolution.append(row_without_spaces['Resolution'])
                self.ref_video.append(row_without_spaces['Ref_video'])
                self.color_encoding.append(row_without_spaces['Color_encoding'])
                self.mos.append(row_without_spaces['Subject_score'])
                self.frame.append(row_without_spaces['frame'])
                self.blockiness.append(row_without_spaces['blockiness'])
                self.spatialactivity.append(row_without_spaces['spatialactivity'])
                self.letterbox.append(row_without_spaces['letterbox'])
                self.pillarbox.append(row_without_spaces['pillarbox'])
                self.blockloss.append(row_without_spaces['blockloss'])
                self.blur.append(row_without_spaces['blur'])
                self.temporalact.append(row_without_spaces['temporalact'])
                self.blockout.append(row_without_spaces['blockout'])
                self.freezing.append(row_without_spaces['freezing'])
                self.exposure.append(row_without_spaces['exposure'])
                self.contrast.append(row_without_spaces['contrast'])
                self.brightness.append(row_without_spaces['brightness'])
                self.interlace.append(row_without_spaces['interlace'])
                self.noise.append(row_without_spaces['noise'])
                self.slice.append(row_without_spaces['slice'])
                self.flickering.append(row_without_spaces['flickering'])
                self.fps.append(row_without_spaces['fps'])
        # print self.name
        # print self.vmaf
        # print self.fps

    def print_attribute(self, attribute):
        print "{} ".format(attribute)


class DBDataAnalyzer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_feature_bar_chart(self, type):
        pass


class AllDBDataAnalyzer(DBDataAnalyzer):
    def __init__(self):
        pass
        self.merged_databases_records = {}
        self.all_fps = []
        self.all_resolutions = []
        self.all_color_encodings = []
        self.all_fps_video_quantity = []


    def add_database(self, database_name , database_obj):
        self.merged_databases_records[database_name] = database_obj
        self.all_fps.extend(database_obj.fps)
        self.all_resolutions.extend(database_obj.resolution)
        self.all_color_encodings.extend(database_obj.color_encoding)


    def create_feature_bar_chart(self, type):
        if "fsp" in type:
            feature_sumup = Counter(self.all_fps)
        elif "resolution" in type:
            feature_sumup = Counter(self.all_resolutions)
        elif "color_encoding" in type:
            feature_sumup = Counter(self.all_color_encodings)
        else:
            print "usage create_feature_bar_chart : fsp, resolution, color_encoding"
            return 0
        feature, video_quantity = zip(*feature_sumup.items())
        video_quantity = [int(counter) for counter in video_quantity]
        # print "fps {} counter {} ".format(feature, video_quantity)
        xs = [i for i, _ in enumerate(feature_sumup)]
        plt.bar(xs, video_quantity)
        plt.ylabel("liczba filmow")
        plt.title(type)

        plt.xticks(xs, feature)
        plt.xlabel(type)

        plt.show()

def create_single_csv(new_csv_name="", *argv):
    for database_path in argv:
        print database_path





class SingleDBDataAnalyzer(DBDataAnalyzer):
    def __init__(self, database):
        self.database_obj = database



    def create_feature_bar_chart(self, type):
        if "fsp" in type:
            feature_sumup = Counter(self.database_obj.fps)
        elif "resolution" in type:
            feature_sumup = Counter(self.database_obj.resolution)
        elif "color_encoding" in type:
            feature_sumup = Counter(self.database_obj.color_encoding)
        else:
            print "usage create_feature_bar_chart : fsp, resolution, color_encoding"
            return 0

        feature, counters = zip(*feature_sumup.items())
        counters = [int(counter) for counter in counters ]
        print "feature {} counter {} ".format(feature, counters)
        xs = [i for i, _ in enumerate(feature_sumup)]
        plt.bar(xs, counters)
        plt.ylabel("liczba filmow")
        plt.title(type)

        plt.xticks(xs, feature)
        plt.xlabel(type)

        plt.show()


def main():
    netflix2_DB = DataObj()
    netflix2_DB.load_data(path=r'C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\netflix2.csv')
    netflix2_DB_analyzer = SingleDBDataAnalyzer(netflix2_DB)
    # netflix2_DB_analyzer.create_feature_bar_chart("fsp")
    # netflix2_DB_analyzer.create_feature_bar_chart("resolution")


    netflix1_DB = DataObj()
    netflix1_DB.load_data(path=r'C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\netflix1.csv')
    netflix1_DB_analyzer = SingleDBDataAnalyzer(netflix1_DB)
    # netflix1_DB_analyzer.create_feature_bar_chart("fsp")

    cif4_DB = DataObj()
    cif4_DB.load_data(path=r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\DB_cif4.csv")
    cif4_DB_analyzer = SingleDBDataAnalyzer(cif4_DB)
    # cif4_DB_analyzer.create_feature_bar_chart("fsp")

    cif_DB = DataObj()
    cif_DB.load_data(path=r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\DB_cif.csv")
    cif_DB_analyzer = SingleDBDataAnalyzer(cif_DB)
    # cif_DB_analyzer.create_feature_bar_chart("fsp")

    DB1 = DataObj()
    DB1.load_data(path=r"data_analysis/csv_bds/DB1.csv")
    DB1_analyzer = SingleDBDataAnalyzer(DB1)
    # DB1_analyzer.create_feature_bar_chart("fsp")

    India_DB = DataObj()
    India_DB.load_data(path=r"C:\Users\elacpol\Desktop\VMAF\csv-ki\final_csv\india_agh.csv")
    India_DB_analyzer = SingleDBDataAnalyzer(India_DB)
    # India_DB_analyzer.create_feature_bar_chart("fsp")



    all_database_analyzer = AllDBDataAnalyzer()
    all_database_analyzer.add_database("netflix2", netflix2_DB)
    # print len(all_database_analyzer.all_color_encodings)
    # print len(all_database_analyzer.all_resolutions), "\n"
    all_database_analyzer.add_database("netflix1", netflix1_DB)
    # print len(all_database_analyzer.all_color_encodings)
    # print len(all_database_analyzer.all_resolutions), "\n"
    all_database_analyzer.add_database("cif4", cif4_DB)
    # print len(all_database_analyzer.all_color_encodings)
    # print len(all_database_analyzer.all_resolutions), "\n"
    all_database_analyzer.add_database("cif",cif_DB)
    # print len(all_database_analyzer.all_color_encodings)
    # print len(all_database_analyzer.all_resolutions), "\n"
    all_database_analyzer.add_database("india", India_DB)
    # print len(all_database_analyzer.all_color_encodings)
    # print len(all_database_analyzer.all_resolutions), "\n"
    # all_database_analyzer.create_feature_bar_chart("fsp")
    # all_database_analyzer.create_feature_bar_chart("resolution")
    # all_database_analyzer.create_feature_bar_chart("color_encoding")