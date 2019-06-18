import openpyxl
import collections
from data_parser import File
import re

class AGHfeatureFile(File):
    def __init__(self):
        self.frame = ''
        self.blockiness = ''
        self.spatialactivity = ''
        self.letterbox = ''
        self.pillarbox = ''
        self.blockloss = ''
        self.blur = ''
        self.temporalact = ''
        self.blockout = ''
        self.freezing = ''
        self.exposure = ''
        self.contrast = ''
        self.brightness = ''
        self.interlace = ''
        self.noise = ''
        self.slice = ''
        self.flickering = ''
        self.fps = ''

    def assing_multiple_values(self,name='', frame = '',blockiness = '',spatialactivity = '',letterbox = '',pillarbox = '',blockloss = '',blur = '',temporalact = '',blockout = '',freezing = '',exposure = '',contrast = '',brightness = '',interlace = '',noise = '',slice = '',flickering = '',fps = '' ):
        self.name = name
        self.frame = frame
        self.blockiness = blockiness
        self.spatialactivity = spatialactivity
        self.letterbox = letterbox
        self.pillarbox = pillarbox
        self.blockloss = blockloss
        self.blur = blur
        self.temporalact = temporalact
        self.blockout = blockout
        self.freezing = freezing
        self.exposure = exposure
        self.contrast = contrast
        self.brightness = brightness
        self.interlace = interlace
        self.noise = noise
        self.slice =  slice
        self.flickering = flickering
        self.fps = fps

    def __str__(self):
        return  "file obj {} , attributes : frame={}, blockiness={}, spatialactivity={}, letterbox={}, pillarbox={}, " \
               "blockloss={}, blur={}, temporalact={}, blockout={}, freezing={}, exposure={}, contrast={}, " \
               "brightness={}, interlace={}, noise={}, slice={}, flickering={}, fps={}".format(                             self.name,
                                                                                                                            self.frame ,
                                                                                                                            self.blockiness ,
                                                                                                                            self.spatialactivity,
                                                                                                                            self.letterbox ,
                                                                                                                            self.pillarbox ,
                                                                                                                            self.blockloss ,
                                                                                                                            self.blur ,
                                                                                                                            self.temporalact ,
                                                                                                                            self.blockout ,
                                                                                                                            self.freezing ,
                                                                                                                            self.exposure ,
                                                                                                                            self.contrast ,
                                                                                                                            self.brightness ,
                                                                                                                            self.interlace ,
                                                                                                                            self.noise ,
                                                                                                                            self.slice ,
                                                                                                                            self.flickering ,
                                                                                                                            self.fps ,
                                                                                                                            )




class VideoFileAghFeatures(object):
    def __init__(self):
        self.agh1 = ''
        self.agh2 = ''
        self.wb = ''
        self.sheet_names = ''
        self.frame = ''
        self.files_database = []
        self.summary_sh = ''
        self.video_quantity = ''


    def load_xlsx(self, path):
        self.wb = openpyxl.load_workbook(path, data_only=True)
        self.sheet_names = self.wb.get_sheet_names()
        print self.sheet_names
        self.video_quantity = len(self.sheet_names)

        self.summary_sh = self.wb[u'summary']

        # print self.summary_sh['C5'].value
        self.__create_files_dataset()






    def __create_files_dataset(self):
        merged_lists = zip(self.summary_sh['A2':'A'+str(self.video_quantity)], self.summary_sh['B2':'B'+str(self.video_quantity)],
                           self.summary_sh['C2':'C' + str(self.video_quantity)], self.summary_sh['D2':'D'+str(self.video_quantity)],
                           self.summary_sh['E2':'E' + str(self.video_quantity)], self.summary_sh['F2':'F'+str(self.video_quantity)],
                           self.summary_sh['G2':'G' + str(self.video_quantity)], self.summary_sh['H2':'H'+str(self.video_quantity)],
                           self.summary_sh['I2':'I' + str(self.video_quantity)], self.summary_sh['J2':'J'+str(self.video_quantity)],
                           self.summary_sh['K2':'K' + str(self.video_quantity)], self.summary_sh['L2':'L'+str(self.video_quantity)],
                           self.summary_sh['M2':'M' + str(self.video_quantity)], self.summary_sh['N2':'N' + str(self.video_quantity)],
                           self.summary_sh['O2':'O' + str(self.video_quantity)],
                           self.summary_sh['P2':'P' + str(self.video_quantity)],
                           self.summary_sh['Q2':'Q' + str(self.video_quantity)],
                           self.summary_sh['R2':'R' + str(self.video_quantity)],
                           )
        file_info = collections.namedtuple('file_info', "raw_filename frame	blockiness	spatialactivity	letterbox	pillarbox	blockloss	blur	temporalact	blackout	freezing	exposure	contrast	brightness	interlace	noise	slice	flickering")
        for raw_filename, frame,	blockiness,	spatialactivity,	letterbox,	pillarbox,	blockloss,	blur,	temporalact,	blackout,	freezing,	exposure,	contrast,	brightness,	interlace,	noise,	slice,	flickering in merged_lists:
            file_obj = AGHfeatureFile()
            name, fps = self.__prepare_filename(raw_filename[0].value)
            file_obj.assing_multiple_values(name, frame[0].value, blockiness[0].value,	spatialactivity[0].value,
                                            letterbox[0].value,	pillarbox[0].value, blockloss[0].value, blur[0].value,
                                            temporalact[0].value, blackout[0].value, freezing[0].value, exposure[0].value,
                                            contrast[0].value, brightness[0].value, interlace[0].value, noise[0].value,
                                            slice[0].value, flickering[0].value, fps)
            self.files_database.append(file_obj)
            print file_obj
        pass

    def __prepare_filename(self,raw_name):  # TV07_1080p30_4_5s_1920x1080_30 - > TV07_1080p30_4_5s
        # print raw_name
        sufix_pattern  = r'(.*)(_\d+x\d+_(\d\d).yuv)'
        match_obj = re.match(sufix_pattern, raw_name)
        filename = match_obj.group(1)
        sufix = match_obj.group(2)
        fps = match_obj.group(3)
        # print filename, " ", sufix, " ", fps,"\n "
        return filename, fps





def main():
    india_vqm = VideoFileAghFeatures()
    # india_vqm.load_xlsx(path='agh_features/india/vqm_1_7.xlsx')
    # india_vqm.load_xlsx(path='agh_features/india/vqm_9_12.xlsx')
    india_vqm.load_xlsx(path='agh_features/online_DB/vqm_cif4.xlsx')

if __name__ == "__main__":
    main()