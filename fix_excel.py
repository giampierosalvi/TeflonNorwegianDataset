import pandas as pd
import openpyxl
import numpy as np
import os
import sys

dir_path = sys.argv[1]
try:
    os.mkdir(dir_path+'_fixed')
except:
    print('directory exists, continuing anyway...')

who = pd.read_excel(dir_path + '/who did what.xlsx')

# no modification here
for word in who['Both']:
    if not isinstance(word, str):
        continue
    # only get the first string in case there are more and take lower case
    word = word.split()[0].lower()
    excel_data = openpyxl.load_workbook(dir_path + '/' + word + '.xlsx')
    excel_data.save(dir_path + '_fixed/' + word + '.xlsx')

# modify the title of the sheet in these cases
for assessor in ['Anne Marte', 'Jeanett']:
    for word in who[assessor]:
        if not isinstance(word, str):
            continue
        # only get the first string in case there are more and take lower case
        word = word.split()[0].lower()
        excel_data = openpyxl.load_workbook(dir_path + '/' + word + '.xlsx')
        sheet = excel_data.worksheets[0]
        sheet.title = assessor
        excel_data.save(dir_path + '_fixed/' + word + '.xlsx')
