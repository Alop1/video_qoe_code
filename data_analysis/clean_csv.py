new_file = open(r"reports/models_summary_csv_new_featuresets_clean.csv", 'w+')
with open(r"reports/models_summary_csv_new_featuresets.csv", 'r+')as f:
    # print f[0]
    # print f[1]
    for id, line in enumerate(f):
        # print line
        splited_line = line.split(',')
        if id == 0:
            new_file.write(line)

        print id, splited_line
        try:
            if id != 0 and float(splited_line[3]):
                new_file.write(line)
        except IndexError:
            print id, " blad ", splited_line[0]
