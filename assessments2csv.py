import glob
import sys
import re
import pandas as pd

dir_path = sys.argv[1]

df = pd.DataFrame()

for filename in glob.glob(dir_path + '/*.xlsx'):
    match = re.search(r'([^/]*).xlsx', filename)
    word = str(match.group(1)) # str is to ensure unicode
    if word == 'who did what':
        continue
    excel_data = pd.read_excel(filename)
    keylist = list(excel_data.keys())
    # find pronunciation start and end column
    pronstart = keylist.index('Score')+1
    pronend = keylist.index('Prosody')
    # get pronunciation into one space separated string
    pronunciation = ' '.join(keylist[pronstart:pronend])
    # get pronunciation scores into one space separated string
    pronScores = excel_data.iloc[:, pronstart].astype(str)
    for col in range(pronstart+1, pronend):
        pronScores += ' ' + excel_data.iloc[:, col].astype(str)
    # dorp pronunciation columns (because of variable number)
    excel_data = excel_data.drop(excel_data.iloc[:, pronstart:pronend], axis=1)
    # add word, pronunciation and pronunciation score columns
    excel_data['Word'] = word
    excel_data['Pronunciation'] = pronunciation
    excel_data['pronScores'] = pronScores
    df = pd.concat((df, excel_data))

df.to_csv(dir_path + '.csv', index=False)
