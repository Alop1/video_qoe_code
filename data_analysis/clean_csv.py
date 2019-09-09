# new_file = open(r"reports/models_summary_csv_new_featuresets_clean.csv", 'w+')
new_file = open(r"reports/czerwiec_5/models_summary_smrLinear_svrLinear_4HL_with_res_clean.csv", 'w+')
with open(r"reports/czerwiec_5/models_summary_smrLinear_svrLinear_4HL_with_res.csv", 'r+')as f:
    # print f[0]
    # print f[1]
    skipped_list = []
    for id, line in enumerate(f):
        # print line
        splited_line = line.split(',')
        if id == 0:
            new_file.write(line)

        print id, splited_line
        try:
            if id != 0 and float(splited_line[3]):
                new_file.write(line)
        except (IndexError,ValueError):
            print id, " blad ", splited_line[0]
            skipped_list.append(id)

print "skipped ", len(skipped_list)