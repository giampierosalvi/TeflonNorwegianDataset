import glob
import sys
import re

dir_path = sys.argv[1]

for filename in glob.glob(dir_path + '/*.xlsx'):
    match = re.search(r'([^/]*).xlsx', filename)
    word = match.group(1)
    if word == 'who did what':
        continue
    excel_data = pd.read_excel(filename)
    keylist = list(excel_data.keys())
    pronstart = keylist.index('Score')+1
    pronend = keylist.index('Prosody')
    pronunciation = ' '.join(keylist[pronstart:pronend])
    # pronScores = [' '.join(els) for els in ]
    for index, row in excel_data.iterrows():
        #pronErrors = ' '.join(keylist[pronstart:pronend])
