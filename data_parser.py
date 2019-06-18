import re, os


class File(object):
    def __init__(self):
        self.name = ''
        self.vmaf = ''
        self.psnr = ''
        self.ssim = ''
        self.ms_ssim = ''
        self.resolution = ''
        self.ref_video = ''
        self.color_encoding = ''
        self.mos = ''
        self.STRRED = ''

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

    def __str__(self):
        return "FILE NAME : {0}, ref: {1},  color encoding : {2}, resolution : {3}, mos : {4}, vmaf: {5}, PSNR :{6}, SSIM {7}, MS_SSIM :{8}, STRRED : {9}".format(self.name,self.ref_video,
                                                                                                                                                   self.color_encoding, self.resolution,
                                                                                                                                                   self.mos, self.vmaf, self.psnr,
                                                                                                                                                   self.ssim, str(self.ms_ssim),self.STRRED)

    def print_agh_featurs(self):
            print  "file obj {} , attributes : frame={}, blockiness={}, spatialactivity={}, letterbox={}, pillarbox={}, " \
                   "blockloss={}, blur={}, temporalact={}, blockout={}, freezing={}, exposure={}, contrast={}, " \
                   "brightness={}, interlace={}, noise={}, slice={}, flickering={}, fps={}".format(self.name,
                                                                                                   self.frame,
                                                                                                   self.blockiness,
                                                                                                   self.spatialactivity,
                                                                                                   self.letterbox,
                                                                                                   self.pillarbox,
                                                                                                   self.blockloss,
                                                                                                   self.blur,
                                                                                                   self.temporalact,
                                                                                                   self.blockout,
                                                                                                   self.freezing,
                                                                                                   self.exposure,
                                                                                                   self.contrast,
                                                                                                   self.brightness,
                                                                                                   self.interlace,
                                                                                                   self.noise,
                                                                                                   self.slice,
                                                                                                   self.flickering,
                                                                                                   self.fps,
                                                                                                   )


    def agh_assing_multiple_values(self,name='', frame = '',blockiness = '',spatialactivity = '',letterbox = '',pillarbox = '',blockloss = '',blur = '',temporalact = '',blockout = '',freezing = '',exposure = '',contrast = '',brightness = '',interlace = '',noise = '',slice = '',flickering = '',fps = '' ):
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
        self.slice = slice
        self.flickering = flickering
        self.fps = fps


class DB1(object):

    def __init__(self,path, csv_name = 'DB1'):
        self.csv_name = csv_name
        self.header = 'Source_file,Ref video,  Resolution, Color_encoding, Aggregate_vmaf, Subject_score, PSNR, SSIM, MS_SSIM\n'
        self.header = 'Source_file,Ref_video,  Resolution, Color_encoding, Aggregate_vmaf, Subject_score, PSNR, SSIM, MS_SSIM, frame ,blockiness , spatialactivity , letterbox , pillarbox, blockloss , blur , temporalact , blockout , freezing , exposure , contrast , brightness , interlace ,noise ,slice , flickering ,fps'
        self.input_files_names = []
        self.files_content = []
        self.pattern = r"(cmd_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_)([yuv]{3}\d\d\d.?)(_vs_)(.+?(?=(\d\d\d\d?x\d\d\d?)([\S]*)( )(.*\d*)))))"
        self.root_dir = path
        self.mos_score_dir = ''
        self.filenames_mos_mapper = []
        self.agh_feature_record_database = []

        self.record_base = []


    def load_vmaf(self):
        self.input_files_names = os.listdir(self.root_dir)
        for file in self.input_files_names:
            with open(self.root_dir+file, 'r+') as f:
                file_content = []
                for line in f:
                    if "cmd" not in line: pass
                    else: file_content.append(line)
                self.files_content.append(file_content)

    def load_mos_score(self, path):
        self.mos_score_dir = path
        filesnames_list = []
        mos_list = []
        with open(self.mos_score_dir , 'r+') as f:
            for line in f:
                mos_score_vector = line.split(',')
                file_name = mos_score_vector[1]
                try:
                    avi_index = file_name.index("avi")
                    file_name = file_name[:avi_index-1]
                except ValueError:pass
                mos = mos_score_vector[30]
                filesnames_list.append(file_name)
                mos_list.append(mos)
        files_mos_list = zip(filesnames_list, mos_list)     # todo rozwazyc slownik
        self.filenames_mos_mapper = files_mos_list[3:]
        return files_mos_list[3:]

    @classmethod
    def _prep_filename(cls, raw_filename): #src_2_hrc7_ -> src02_hrc07
        # print raw_filename
        hrc_pattern = r'.*(hrc)(\d)$'
        src_pattern = r'(src_(\d)_)'
        try:
            src_match = re.findall(src_pattern, raw_filename)[0][1]
            # print "1 src_match ", src_match
            raw_filename = re.sub(r'src_\d', r'src0' + src_match, raw_filename)
            # print "ready ", raw_filename
        except IndexError:
            try:
                two_digits_pattern = r'(src_(\d\d)_)'
                src_match = re.findall(two_digits_pattern, raw_filename)[0][1]
                raw_filename = re.sub(r'src_\d\d', r'src' + src_match, raw_filename)
                # print raw_filename
            except Exception as e:
                # print  "1 " , e
                pass
        try:
            match_obj = re.findall(hrc_pattern, raw_filename)[0][1]
            # print "match ", match_obj
            ready_filename = re.sub(r'hrc\d', r'hrc0'+match_obj, raw_filename)
            # print "ready ", ready_filename
            # print "return ready ",ready_filename
            return ready_filename
        except Exception as e :
            # print "return raw", raw_filename
            return raw_filename

    def load_features(self, path, psnr=0, ssim=0, ms_ssim=0 ):
        with open(path, 'r+') as f:
            name_pattern = r'(example_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_)([yuv]{3}\d\d\d.?)(_vs_)(.+?(?=(\d\d\d\d?x\d\d\d?)))))'
            feature_pattern = r'.*:(\d+\.?\d*)'
            is_add_new_record = False
            name, feature_score, pattern_prefix = '', '', ''
            for line in f:
                print "[line]", line
                if line.startswith('example') : pattern_prefix = 'example'
                elif line.startswith('dataset') : pattern_prefix = 'dataset'
                elif line.startswith("india"): pattern_prefix = 'india'
                if pattern_prefix:
                    print "[patter prefix]"
                    name_pattern = r'('+ pattern_prefix+ r'_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_)([yuv]{3}\d\d\d.?)(_vs_)(.+?(?=(\d\d\d\d?x\d\d\d?)))))'
                    print name_pattern
                    match_obj = re.match(name_pattern,line)
                    name = match_obj.group(7)
                    if name[-1] == '_':
                        name = name[:-1]
                    name = self._prep_filename(name)
                    pattern_prefix= ''
                elif line.startswith('Aggregate'):
                    print "in aggregate"
                    match_obj = re.match(feature_pattern, line)
                    feature_score = match_obj.group(1)
                    is_add_new_record = True
                if is_add_new_record:
                    for file_obj in self.record_base:
                        if file_obj.name == name:
                            if psnr:
                                file_obj.psnr = feature_score
                            elif ssim:
                                file_obj.ssim = feature_score
                            elif ms_ssim:
                                file_obj.ms_ssim = feature_score
                            print file_obj
                    is_add_new_record = False




    def get_info(self, line):
        match_obj = re.match(self.pattern, line)
        # print line
        try:
            aggregate_vmaf, color_encoding, ref_file, resolution, src_file = self.extract_base_feature(
                match_obj)
        except AttributeError:
            pattern = r"(cmd_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_vs_)(.+?(?=(\d\d\d\d?x\d\d\d?)([\S]*)( )(.*\d*)))))"
            match_obj = re.match(pattern, line)
            src_file = match_obj.group(5)
            src_file = src_file if src_file[-1] != '_' else src_file[:-1]
            src_file = self._prep_filename(src_file)
            ref_file = match_obj.group(2)
            resolution = match_obj.group(3)
            color_encoding = " I420"
            aggregate_vmaf = match_obj.group(9)
            print "except"

        mos = self._match_mos_to_file_name(src_file)

        file_obj = self.create_file_object(aggregate_vmaf, color_encoding, mos, ref_file, resolution,
                                           src_file)
        # print file_obj
        return file_obj

    def extract_base_feature(self, match_obj):
        file_obj = File()
        src_file = match_obj.group(7)
        src_file = src_file if src_file[-1] != '_' else src_file[:-1]
        src_file = self._prep_filename(src_file)
        ref_file = match_obj.group(2)
        resolution = match_obj.group(3)
        color_encoding = match_obj.group(5)
        aggregate_vmaf = match_obj.group(11)
        # print "[extract_base_feature] {}, {}, {}, {}, {}".format(aggregate_vmaf, color_encoding, ref_file, resolution, src_file)
        return aggregate_vmaf, color_encoding, ref_file,  resolution, src_file

    def create_file_object(self, aggregate_vmaf, color_encoding,  mos, ref_file, resolution, src_file):
        file_obj = File()
        file_obj.name = src_file
        print "name ", file_obj.name
        file_obj.resolution = resolution
        file_obj.ref_video = ref_file
        file_obj.color_encoding = color_encoding
        file_obj.vmaf = aggregate_vmaf
        file_obj.mos = mos

        return file_obj

    def create_base_records(self):       #todo zmienic nazwe
            # print "[create_base_records]"
            for file in self.files_content:
                # print "[create_base_records] for 1 "
                for line in file:
                    # print "[create_base_records] for 2 :", line
                    file_obj = self.get_info(line)
                    # print "[create_base_records]", file_obj.name
                    self.record_base.append(file_obj)


    def create_csv_final(self, path):
        with open(path, 'w') as f:
            f.write(self.header)#todo zmienic header
            for file_obj in self.record_base:
                # print "new info psnr {}, sime: {}, ms_ssim : {}".format(file_obj.psnr,file_obj.ssim, file_obj.ms_ssim)
                print file_obj
                file_obj.print_agh_featurs
                csv_input = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {},{}, {}, {} ,{}, {}, {}, {}, {}\n"\
                                                                        .format(file_obj.name, file_obj.ref_video,
                                                                          file_obj.resolution, file_obj.color_encoding,
                                                                          file_obj.vmaf, file_obj.mos, file_obj.psnr,
                                                                          file_obj.ssim, file_obj.ms_ssim, file_obj.frame,
                                                                          file_obj.blockiness, file_obj.spatialactivity,
                                                                          file_obj.letterbox, file_obj.pillarbox,file_obj.blockloss,
                                                                          file_obj.blur, file_obj.temporalact,
                                                                          file_obj.blockout, file_obj.freezing, file_obj.exposure,
                                                                          file_obj.contrast, file_obj.brightness,
                                                                          file_obj.interlace, file_obj.noise, file_obj.slice,
                                                                          file_obj.flickering, file_obj.fps)
                f.write(csv_input)

    def _match_mos_to_file_name(self, file_name):
        try:
            # print "[match mos] filename: ", file_name
            temp_mapper = dict(self.filenames_mos_mapper)
            mos_score = temp_mapper[file_name]
            return mos_score
        except KeyError:
            return "none"


class DB_italy_switz(DB1):
    # info: dane poprawnie zebrane
    def get_info(self, line):
        match_obj, src_file = self.method_name(line)
        # print "[get info ] 1 srcfile", src_file
        src_file = self._prep_filename(src_file)
        # print "[get info ] 2 srcfile", src_file
        ref_file = match_obj.group(2)
        resolution = match_obj.group(3)
        color_encoding = " I420"
        aggregate_vmaf = match_obj.group(9)
        mos = self._match_mos_to_file_name(src_file)

        file_obj = File()
        file_obj.name = src_file
        file_obj.resolution = resolution
        file_obj.ref_video = ref_file
        file_obj.color_encoding = color_encoding
        file_obj.vmaf = aggregate_vmaf
        file_obj.mos = mos
        return file_obj

    def method_name(self, line):
        self.pattern = r"(cmd_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_vs_)(.+?(?=(\d\d\d\d?x\d\d\d?)([\S]*)( )(.*\d*)))))"
        match_obj = re.match(self.pattern, line)
        src_file = match_obj.group(5)
        src_file = src_file if src_file[-1] != '_' else src_file[:-1]
        return match_obj, src_file

    def load_mos_score(self, path):
        self.mos_score_dir = path
        filesnames_list = []
        mos_list = []
        with open(self.mos_score_dir, 'r+') as f:
            for line in f.readlines()[1:]:
                mos_score_vector = line.split(',')
                file_name = mos_score_vector[0]
                mos = mos_score_vector[41]
                mos  = mos.replace('\n', '')
                file_name.replace("'", '')
                dash_digit_dash_pattern = r'_\d\d_'
                file_name = re.sub(dash_digit_dash_pattern, r'_', file_name)
                print "[load mos] ", file_name, ': ', mos
                filesnames_list.append(file_name)
                mos_list.append(mos)
        files_mos_list = zip(filesnames_list, mos_list)  # todo rozwazyc slownik
        self.filenames_mos_mapper = files_mos_list[3:]
        return files_mos_list

    @classmethod
    def _prep_filename(cls, raw_filename):
        raw_filename.replace("'", '')
        print raw_filename
        dash_digit_dash_pattern = r'_\d\d?_'
        raw_filename = re.sub(dash_digit_dash_pattern, r'_', raw_filename)
        if raw_filename[-1] == '_':
            raw_filename = raw_filename[:-1]
        # if 'plr' not in raw_filename:
        #     raw_filename = raw_filename + '_plr0_0'
        return raw_filename

    def load_features(self, path, psnr=0, ssim=0, ms_ssim=0 ):
        with open(path, 'r+') as f:
            global cif_type
            print cif_type
            name_pattern = r'(Italy_switz_' + cif_type + r'_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_vs_)(.+?(?=(\d\d\d\d?x\d\d\d?)))))'
            if ms_ssim:
                feature_pattern = r'.*MS_SSIM_score:(\d+\.?\d*)'
            elif psnr:
                feature_pattern = r'.*PSNR_score:(\d+\.?\d*)'
            elif ssim:
                feature_pattern =  r'.*SSIM_score:(\d+\.?\d*)'
            is_add_new_record = False
            name, feature_score = '', ''
            for line in f:
                if line.startswith('Italy_switz'):
                    match_obj = re.match(name_pattern,line)
                    name = match_obj.group(5)
                    if name[-1] == '_':
                        name = name[:-1]
                    name = self._prep_filename(name)
                    # print "[load feature] ", name
                elif line.startswith('Aggregate'):
                    match_obj = re.match(feature_pattern, line)
                    feature_score = match_obj.group(1)
                    # print "feature pattern ", feature_score
                    is_add_new_record = True
                if is_add_new_record:
                    for file_obj in self.record_base:
                        if file_obj.name == name:
                            if psnr:
                                file_obj.psnr = feature_score
                            elif ssim:
                                file_obj.ssim = feature_score
                            elif ms_ssim:
                                file_obj.ms_ssim = feature_score
                    is_add_new_record = False


class Netflix2(DB1):
    def load_vmaf(self):
        self.input_files_names = os.listdir(self.root_dir)
        for file in self.input_files_names:
            print "[load_files_content] ", file[:-4]
            with open(self.root_dir+file, 'r+') as f:
                file_content = []
                for line in f:
                    new_line = line.replace("\n", '')
                    file_content.append(new_line)
                self.files_content.append((file[:-4],file_content))

    def load_mos_score(self, path):
        self.mos_score_dir = path
        mos_files = os.listdir(self.mos_score_dir)
        for file in mos_files:
            print "[load mos] ", file[:-8]
            with open(self.mos_score_dir+file, 'r+') as f:
                file_content = []
                for line in f:
                    line.replace(r"\n", '')
                    file_content.append(line)
                self.filenames_mos_mapper.append((file[:-8],file_content))

    def create_base_records(self):
        # with open(self.csv_name+'.csv', 'w') as f:
        #     f.write(self.header)
            for filename, vmaf in self.files_content:
                    vmaf_str_list = vmaf[0].split(',')
                    vmaf_score_list = [float(ele) for ele in vmaf_str_list]
                    vmaf_average = sum(vmaf_score_list)/len(vmaf_score_list)
                    resolution = '1920x1080'
                    color_encoding = '420'
                    mos_string = self._match_mos_to_file_name(filename)
                    mos_vect = mos_string[0].split(',')
                    mos_vect = [float(ele) for ele in mos_vect]
                    mos = sum(mos_vect) / len(mos_vect)

                    file_obj = File()
                    file_obj.name = filename
                    file_obj.resolution = resolution
                    file_obj.ref_video = '---'
                    file_obj.color_encoding = color_encoding
                    file_obj.vmaf = vmaf_average
                    file_obj.mos = mos
                    self.record_base.append(file_obj)
                    print file_obj
                    # csv_input = "{}, {}, {}, {}, {}, {}\n".format(filename, "---", resolution, color_encoding, vmaf_average, mos)
                    # print csv_input
                    # f.write(csv_input)

    def _match_mos_to_file_name(self, file_name):
        temp_dict = dict(self.filenames_mos_mapper)
        try:
            return temp_dict[file_name]
        except KeyError:
            return 'none'

    @staticmethod
    def _prep_filename(raw_filename, psnr=0, ssim=0, ms_ssim=0, STRRED = 0):
        # print raw_filename
        if psnr:
            raw_filename = raw_filename.replace("_psnr", '')
        elif ssim:
            raw_filename = raw_filename.replace("_ssim", '')
        elif ms_ssim:
            raw_filename = raw_filename.replace('_ms_ssim', '')
        elif STRRED:
            raw_filename = raw_filename.replace('_STRRED', '')
        filename = raw_filename[:-4]
        # print "file_name ", filename
        return filename

    def _is_vector(self,file_content):
        file_content_tab = file_content.split(",")
        if len(file_content) > 1: return True
        else: return False

    def _get_final_score(self, line):
        feature_vec = line.split(",")
        # print feature_vec
        feature_vec = [float(ele) for ele in feature_vec]
        final_score =  sum(feature_vec)/len(feature_vec)
        # print "suma {}, dlugosc: {}, srednia : {}".format(sum(feature_vec), len(feature_vec), sum(feature_vec)/len(feature_vec))
        return str(final_score)


    def load_features(self, path, psnr=0, ssim=0, ms_ssim=0, STRRED = 0):
        self.feature_score_dir = path
        feature_files = os.listdir(self.feature_score_dir)
        feature_score = ''
        for file in feature_files:
            with open(self.feature_score_dir + file, 'r+') as f:
                for line in f:
                    line.replace(r"\n", '')
                    if self._is_vector(line):
                        feature_score = self._get_final_score(line)
                    else:
                        feature_score = line
                    file_name = Netflix2._prep_filename(file, psnr, ssim, ms_ssim, STRRED)
                    # print file_name, "  content : ", feature_score
                    for file_obj in self.record_base:
                        # print file_name, "  content : ", feature_score, "from class name: ", file_obj.name
                        if file_obj.name == file_name:
                            # print "get"
                            if psnr:
                                file_obj.psnr = feature_score
                            elif ssim:
                                file_obj.ssim = feature_score
                            elif ms_ssim:
                                file_obj.ms_ssim = feature_score
                            elif STRRED:
                                file_obj.STRRED = feature_score
                            # print file_obj
                            break


                # self.filenames_mos_mapper.append((file[:-8], file_content))


class Netflix1(Netflix2):
    @classmethod
    def _prep_filename(cls, raw_filename):
        cont_pattern = r'cont'
        raw_filename = re.sub(cont_pattern, 'content', raw_filename)
        return raw_filename

    def load_vmaf(self):
        self.input_files_names = os.listdir(self.root_dir)
        for file in self.input_files_names:
            with open(self.root_dir+file, 'r+') as f:
                file_content = []
                for line in f:
                    if "cmd" not in line: pass
                    else: file_content.append(line)
                self.files_content.append(file_content)

    # def load_vmaf(self):
    #     """dane tylko z pliku zrodowego - nieliczonych """
    #     self.input_files_names = os.listdir(self.root_dir)
    #     for file in self.input_files_names:
    #         with open(self.root_dir + file, 'r+') as f:
    #             file_content = []
    #             for line in f:
    #                 if not line : continue
    #                 file_content.append(line)
    #             file = file[:-10]
    #             vmaf_mos = file_content[0].split(",")
    #             # print "file: ", file,"vmaf/mos:  ", file_content[0], "vmaf: ",  vmaf_mos[0], "mos: ",  vmaf_mos[1]
    #             vmaf  = vmaf_mos[0]
    #             mos = vmaf_mos[1]
    #             self.files_content.append((file,vmaf, mos))

    def load_mos_score(self, path):
        # print "not avaliable"
        self.mos_score_dir = path
        mos_files = os.listdir(self.mos_score_dir)
        for file in mos_files:
            with open(self.mos_score_dir + file, 'r+') as f:
                file_content = []
                for line in f:
                    if not line : continue
                    file_content.append(line)
                file = file[:-10]
                vmaf_mos = file_content[0].split(",")
                # print "file: ", file,"vmaf/mos:  ", file_content[0], "vmaf: ",  vmaf_mos[0], "mos: ",  vmaf_mos[1]
                # vmaf  = vmaf_mos[0]
                mos = vmaf_mos[1].replace("\n", '')
                self.filenames_mos_mapper.append((file, mos))

    def create_base_records_old(self):
        with open(self.csv_name+'.csv', 'w') as f:
            f.write(self.header)
            for file in self.files_content:
                for line in file:
                    info = self.get_info(line)
                    # print info
                    csv_input = "{}, {}, {}, {}, {}, {}\n".format(*info)
                    f.write(csv_input)

    def create_base_records(self):
        # with open(self.csv_name+'.csv', 'w') as f:
        #     f.write(self.header)
        # print self.files_content
        for file in self.files_content:
            for line in file:
                # print line
                file_obj = self.get_info(line)
                self.record_base.append(file_obj)

    # def create_csv(self):
    #     """dla danych tylk od zrodla- nie liczonych """
    #     with open(self.csv_name+'.csv', 'w') as f:
    #         f.write(self.header)
    #         for filename, vmaf, mos in self.files_content:
    #                 resolution = '1920x1080'
    #                 color_encoding = '420'
    #                 csv_input = "{}, {}, {}, {}, {}, {}\n".format(filename, "---", resolution, color_encoding, vmaf, mos)
    #                 print csv_input
    #                 f.write(csv_input)


class India(DB1):
    # def get_info(self, line):
    #     self.pattern = r"(cmd_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_vs_)(.+?(?=(\d\d\d\d?x\d\d\d?)([\S]*)( )(.*\d*)))))"
    #     match_obj = re.match(self.pattern, line)
    #     src_file = match_obj.group(5)
    #     src_file = src_file if src_file[-1] != '_' else src_file[:-1]
    #     # print "[vmaf name ]",src_file
    #     # print "[get info ] 1 srcfile", src_file
    #     src_file = self._prep_filename(src_file)
    #     # print "[get info ] 2 srcfile", src_file
    #     ref_file = match_obj.group(2)
    #     resolution = match_obj.group(3)
    #     color_encoding = " I420"
    #     aggregate_vmaf = match_obj.group(9)
    #     mos = self._match_mos_to_file_name(src_file)
    #     return (src_file, ref_file, resolution, color_encoding, aggregate_vmaf, mos)

    def load_mos_score(self, path):
        filesnames_list = []
        mos_list = []
        self.mos_score_dir = path
        mos_files = os.listdir(self.mos_score_dir)
        for file in mos_files:
            with open(self.mos_score_dir+file , 'r+') as f:
                # print file
                for line in f:
                    try:
                        avi_index = file.index("_mos.csv")
                        ready_file_name = file[:avi_index]
                        # print "[load mos] ", ready_file_name
                    except ValueError:print "except"
                    mos = line.replace("\n", '')
                    filesnames_list.append(ready_file_name)
                    mos_list.append(mos)
            files_mos_list = zip(filesnames_list, mos_list)     # todo rozwazyc slownik
            self.filenames_mos_mapper = files_mos_list
        # print "[load mos mapper ] ", self.filenames_mos_mapper
        return files_mos_list

    def get_info(self, line):       #ta sama co w calsie rodzic nie trzeba nadpisywac
        match_obj = re.match(self.pattern, line)
        try:
            aggregate_vmaf, color_encoding, file_obj, ref_file, resolution, src_file = self.extract_base_feature(
                match_obj)
            print "try"
        except AttributeError:
            pattern = r"(cmd_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_vs_)(.+?(?=(\d\d\d\d?x\d\d\d?)([\S]*)( )(.*\d*)))))"
            match_obj = re.match(pattern, line)
            src_file = match_obj.group(5)
            src_file = src_file if src_file[-1] != '_' else src_file[:-1]
            src_file = self._prep_filename(src_file)
            ref_file = match_obj.group(2)
            resolution = match_obj.group(3)
            color_encoding = " I420"
            aggregate_vmaf = match_obj.group(9)

        mos = self._match_mos_to_file_name(src_file)

        file_obj = self.create_file_object(aggregate_vmaf, color_encoding, mos, ref_file, resolution,
                                           src_file)
        # print file_obj

        return file_obj


    def load_features(self, path, psnr=0, ssim=0, ms_ssim=0 ):
        with open(path, 'r+') as f:
            feature_pattern = r'.*:(\d+\.?\d*)'
            is_add_new_record = False
            name, feature_score, pattern_prefix = '', '', ''
            for line in f:
                if line.startswith('example') : pattern_prefix = 'example'
                elif line.startswith('dataset') : pattern_prefix = 'dataset_india'
                elif line.startswith("india"): pattern_prefix = 'india'
                if pattern_prefix:
                    name_pattern = r'('+ pattern_prefix+ r'_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_vs_)(.+?(?=(_\d\d\d\d?x\d\d\d?)))).*)'
                    # print line
                    # print name_pattern
                    match_obj = re.match(name_pattern,line)
                    try:
                        name = match_obj.group(5)
                    except AttributeError:# try another pattern
                        name_pattern = r'(' + pattern_prefix + r'_\d+_\d+_)(.+?(?=(\d\d\d\d?x\d\d\d\d?)(_)([yuv]{3}\d\d\d.?)(_vs_)(.+?(?=(_\d\d\d\d?x\d\d\d?)))).*)'
                        # print line
                        # print name_pattern
                        match_obj = re.match(name_pattern, line)
                        name = match_obj.group(7)
                    if name[-1] == '_':
                        name = name[:-1]
                    name = self._prep_filename(name)
                    pattern_prefix = ''
                elif line.startswith('Aggregate'):
                    match_obj = re.match(feature_pattern, line)
                    feature_score = match_obj.group(1)
                    is_add_new_record = True
                if is_add_new_record:
                    for file_obj in self.record_base:
                        if file_obj.name == name:
                            if psnr:
                                file_obj.psnr = feature_score
                            elif ssim:
                                file_obj.ssim = feature_score
                            elif ms_ssim:
                                file_obj.ms_ssim = feature_score
                            print file_obj
                    is_add_new_record = False


class Toni(DB1):
    def load_mos_score(self, path):
        self.mos_score_dir = path
        filesnames_list = []
        mos_list = []
        with open(self.mos_score_dir, 'r+') as f:
            for line in f:
                mos_score_vector = line.split(',')
                file_name = mos_score_vector[1]
                try:
                    avi_index = file_name.index("avi")
                    file_name = file_name[:avi_index - 1]
                except ValueError:
                    pass
                mos = mos_score_vector[2]
                filesnames_list.append(file_name)
                mos_list.append(mos)
        files_mos_list = zip(filesnames_list, mos_list)  # todo rozwazyc slownik
        self.filenames_mos_mapper = files_mos_list
        # print "[load mos] ",  self.filenames_mos_mapper
        return files_mos_list


def merge_databases(base_database, merged_database):
    for base_obj in base_database:

        for merged_obj in merged_database:
            print "[base obj ] {} merged_obj : {}" .format(base_obj.name, merged_obj.name)
            if base_obj.name == merged_obj.name:
                print "hit"
                base_obj.agh_assing_multiple_values(merged_obj.name, merged_obj.frame, merged_obj.blockiness,	merged_obj.spatialactivity,
                                                    merged_obj.letterbox,	merged_obj.pillarbox, merged_obj.blockloss, merged_obj.blur,
                                                    merged_obj.temporalact, merged_obj.blockout, merged_obj.freezing, merged_obj.exposure,
                                                    merged_obj.contrast, merged_obj.brightness, merged_obj.interlace, merged_obj.noise,
                                                    merged_obj.slice, merged_obj.flickering, merged_obj.fps)
                break

cif_type = ''
def main():
    pass
    import VideoFileAghFeatures
    # path  = "DB1/VMAF_parsed/"
    # data_set_DB1 = DB1(path)
    # data_set_DB1.load_vmaf()
    # data_set_DB1.load_mos_score(path="DB1/IRCCyN_IVC_SVC4QoE_QP0_QP1_Database_Score.csv")
    # data_set_DB1.create_base_records()
    # data_set_DB1.load_features(path='other_features/psnr_without_33.txt', psnr=True)
    # data_set_DB1.load_features(path='other_features/ssim_without_33', ssim=True)
    # data_set_DB1.load_features(path='other_features/ms_ssim_all_results', ms_ssim=True)
    #
    # DB1_vqm = VideoFileAghFeatures.VideoFileAghFeatures()
    # DB1_vqm.load_xlsx(path='agh_features/DB1/vqm_DB1.xlsx')
    # for file_obj in DB1_vqm.files_database:
    #
    #     new_name = DB1._prep_filename(file_obj.name)
    #     # print "old name {}, new name {}".format(file_obj.name, new_name)
    #     file_obj.name = new_name
    #     # print "new name in obj {}".format(file_obj.name)
    #     # file_obj.print_agh_featurs
    # print "END"
    #
    # merge_databases(base_database=data_set_DB1.record_base, merged_database=DB1_vqm.files_database)
    # data_set_DB1.create_csv_final(path='csv-ki/final_csv/DB1.csv')



    path2 = "VMAF_BD2_italy_switz/vmaf_parsed/cif4/"
    global cif_type
    # cif_type = 'cif4'
    # data_set_italy_switz = DB_italy_switz(path=path2)
    # data_set_italy_switz.load_vmaf()
    # data_set_italy_switz.load_mos_score(path="VMAF_BD2_italy_switz/b_RawSubjData/4CIF/subject_cif4.csv")
    # data_set_italy_switz.create_base_records()
    # data_set_italy_switz.load_features(path='other_features/italy_switz/ms_ssim_cif4.txt', ms_ssim=True)
    # data_set_italy_switz.load_features(path='other_features/italy_switz/psnr_cif4.txt', psnr=True)
    # data_set_italy_switz.load_features(path='other_features/italy_switz/ssim_cif4.txt', ssim=True)
    # dataset_cif4 = VideoFileAghFeatures.VideoFileAghFeatures()
    # dataset_cif4.load_xlsx(path='agh_features/online_DB/vqm_cif4.xlsx')
    # for file_obj in dataset_cif4.files_database:
    #
    #     new_name = DB_italy_switz._prep_filename(file_obj.name)
    #     # print "old name {}, new name {}".format(file_obj.name, new_name)
    #     file_obj.name = new_name
    #     # print "new name in obj {}".format(file_obj.name)
    #     # file_obj.print_agh_featurs
    # print "END"
    # #
    # merge_databases(base_database=data_set_italy_switz.record_base, merged_database=dataset_cif4.files_database)
    # data_set_italy_switz.create_csv_final(path="csv-ki/final_csv/DB_cif4.csv")

    # path2_2 = "VMAF_BD2_italy_switz/vmaf_parsed/cif/"
    # cif_type = 'cif'
    # data_set_italy_switz = DB_italy_switz(path=path2_2)
    # data_set_italy_switz.load_vmaf()
    # data_set_italy_switz.load_mos_score(path="VMAF_BD2_italy_switz/b_RawSubjData/CIF/subject_cif.csv")
    # data_set_italy_switz.create_base_records()
    # data_set_italy_switz.load_features(path='other_features/italy_switz/ms_ssim_cif.txt', ms_ssim=True)
    # data_set_italy_switz.load_features(path='other_features/italy_switz/psnr_cif.txt', psnr=True)
    # data_set_italy_switz.load_features(path='other_features/italy_switz/ssim_cif.txt', ssim=True)
    # dataset_cif = VideoFileAghFeatures.VideoFileAghFeatures()
    # dataset_cif.load_xlsx(path='agh_features/online_DB/vqm_cif.xlsx')
    # for file_obj in dataset_cif.files_database:
    #     new_name = DB_italy_switz._prep_filename(file_obj.name)
    #     # print "old name {}, new name {}".format(file_obj.name, new_name)
    #     file_obj.name = new_name
    #     # print "new name in obj {}".format(file_obj.name)
    #     # file_obj.print_agh_featurs
    # # print "END"
    # merge_databases(base_database=data_set_italy_switz.record_base, merged_database=dataset_cif.files_database)
    # data_set_italy_switz.create_csv_final(path="csv-ki/final_csv/DB_cif.csv")
    #
    # path3 = r"C:/Users/elacpol/Desktop/VMAF/VMAF_netflix_2/vamf/"
    # data_set_netflix_2 = Netflix2(path=path3, csv_name="netflix2_new")
    # data_set_netflix_2.load_vmaf()
    # data_set_netflix_2.load_mos_score(path="VMAF_netflix_2/mos/")
    # data_set_netflix_2.create_base_records()
    # data_set_netflix_2.load_features(path="C:\Users\elacpol\Desktop\VMAF\VMAF_netflix_2\other_features\psnr\\", psnr=True)
    # data_set_netflix_2.load_features(path="C:\Users\elacpol\Desktop\VMAF\VMAF_netflix_2\other_features\ssim\\", ssim=True)
    # data_set_netflix_2.load_features(path="C:\Users\elacpol\Desktop\VMAF\VMAF_netflix_2\other_features\ms_ssim\\", ms_ssim=True)
    # data_set_netflix_2.load_features(path="C:\Users\elacpol\Desktop\VMAF\VMAF_netflix_2\other_features\STRRED\\", STRRED=True)
    # datasate_agh_netflix2 = VideoFileAghFeatures.VideoFileAghFeatures()
    # datasate_agh_netflix2.load_xlsx(path='agh_features/netflix2/vqm_netflix2_part1.xlsx')
    # datasate_agh_netflix2.load_xlsx(path='agh_features/netflix2/vqm_netflix2_part2.xlsx')
    # datasate_agh_netflix2.load_xlsx(path='agh_features/netflix2/vqm_netflix2_part3.xlsx')
    # merge_databases(base_database=data_set_netflix_2.record_base, merged_database=datasate_agh_netflix2.files_database)
    # data_set_netflix_2.create_csv_final(path="csv-ki/final_csv/netflix2.csv")


    path4 = "VMAF_netflix_1/vmaf_outputs/passerd/"
    data_set_netflix_1 = Netflix1(path=path4, csv_name='netflix1')
    data_set_netflix_1.load_vmaf()
    data_set_netflix_1.load_mos_score(path="VMAF_netflix_1/vmaf_outputs/mos/")
    data_set_netflix_1.create_base_records()
    data_set_netflix_1.load_features(path="VMAF_netflix_1/other_features/vec/psnr/", psnr=True)
    data_set_netflix_1.load_features(path="VMAF_netflix_1/other_features/vec/ssim/", ssim=True)
    data_set_netflix_1.load_features(path="VMAF_netflix_1/other_features/vec/ms_ssim/", ms_ssim=True)
    # data_set_netflix_1.load_features(path="VMAF_netflix_1/other_features/vec/STRRED/", STRRED=True)
    #todo dododac niqe
    #todo dodac gmsd
    datasate_agh_netflix1 = VideoFileAghFeatures.VideoFileAghFeatures()
    datasate_agh_netflix1.load_xlsx(path='agh_features/netflix1/vqm_netflix1.xlsx')
    for file_obj in datasate_agh_netflix1.files_database:
        new_name = Netflix1._prep_filename(file_obj.name)
            # print "old name {}, new name {}".format(file_obj.name, new_name)
        file_obj.name = new_name
            # print "new name in obj {}".format(file_obj.name)
            # file_obj.print_agh_featurs
    merge_databases(base_database=data_set_netflix_1.record_base, merged_database=datasate_agh_netflix1.files_database)
    data_set_netflix_1.create_csv_final(path="csv-ki/final_csv/netflix1.csv")


    # path5 = 'VMAF_Indie/parssed/'
    # data_set_india = India(path=path5)
    # data_set_india.load_vmaf()
    # data_set_india.load_mos_score(path="VMAF_Indie/mos/")
    # data_set_india.create_base_records()
    # data_set_india.load_features(path="C:\Users\elacpol\Desktop\VMAF\VMAF_Indie\other_features\PSNR.txt",psnr=True)
    # data_set_india.load_features(path='VMAF_Indie/other_features/SSIM.txt', ssim=True)
    # data_set_india.load_features(path='VMAF_Indie/other_features/MS_SSIM.txt', ms_ssim=True)
    #
    # india_vqm = VideoFileAghFeatures.VideoFileAghFeatures()
    # india_vqm.load_xlsx(path='agh_features/india/vqm_1_7.xlsx')
    # india_vqm.load_xlsx(path='agh_features/india/vqm_9_12.xlsx')
    # india_vqm.load_xlsx(path='agh_features/india/vqm_13_16.xlsx')
    # india_vqm.load_xlsx(path='agh_features/india/vqm_17_18.xlsx')
    # india_vqm.load_xlsx(path='agh_features/india/vqm_failed1.xlsx')
    # india_vqm.load_xlsx(path='agh_features/india/vqm_failed2.xlsx')
    # merge_databases(base_database=data_set_india.record_base, merged_database=india_vqm.files_database)
    # data_set_india.create_csv_final(path="csv-ki/final_csv/india_agh.csv")



    #zla baza
    # path6 = "VMAF_fomToni/parsed/"
    # data_set_toni = Toni(path=path6, csv_name="toni")
    # data_set_toni.load_files_content()
    # data_set_toni.load_mos_score(path="VMAF_fomToni/Realignment_MOS_comma.csv")
    # data_set_toni.create_csv()

if __name__ == "__main__":
    main()