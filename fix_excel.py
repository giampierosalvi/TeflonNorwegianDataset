import pandas as pd
import openpyxl
import numpy as np

who = pd.read_excel('assessments2023-03-29/who did what.xlsx')

# no modification here
for word in who['Both']:
    if not isinstance(word, str):
        continue
    # only get the first string in case there are more and take lower case
    word = word.split()[0].lower()
    excel_data = openpyxl.load_workbook('assessments2023-03-29/' + word + '.xlsx')
    excel_data.save('assessments2023-03-09mod/' + word + '.xlsx')

# modify the title of the sheet in these cases
for assessor in ['Together', 'Anne Marte', 'Jeanett']:
    for word in who[assessor]:
        if not isinstance(word, str):
            continue
        # only get the first string in case there are more and take lower case
        word = word.split()[0].lower()
        excel_data = openpyxl.load_workbook('assessments2023-03-29/' + word + '.xlsx')
        sheet = excel_data.worksheets[0]
        sheet.title = assessor
        excel_data.save('assessments2023-03-09mod/' + word + '.xlsx')
