filename = "kjdhakdjha.avi       "
indx = filename.index(".avi")
print indx


import os
import pandas as pd
import openpyxl

wb = openpyxl.load_workbook("testing_workbook.xlsx", data_only=True)
sheet_names = wb.get_sheet_names()
print sheet_names
sh1 = wb[u'Sheet3']
print sh1['D4'].value


def validateData(dataFile):
    import datetime
    """
    Return True if time in minutes between printouts is less that value defined in "wait" variable
    :param dataFile:
    :return:
    """
    fD = {}
    wait = 12
    lastDate = ''
    for line in dataFile:
        line = line.split(",")
        if len(line) > 4:
            validate = False
            lastDate = str(line[4])

    print("'" + lastDate + "'")
    lastDate = datetime.datetime.strptime(lastDate, '%Y-%m-%d %H:%M:%S.%f ')
    #     currentDate = datetime.datetime.now()
    print(lastDate)
    if not True:
        dataFile.close()
        return False

    return True

t = 'Documents/'
a = open('kotek.html', 'a+')
validateData(a)
a.close()