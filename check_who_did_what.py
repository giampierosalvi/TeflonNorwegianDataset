import pandas as pd
import openpyxl
import sys
import glob

dir_path = sys.argv[1]

who = pd.read_excel(dir_path + '/who did what.xlsx')

def duplicates(a):
    seen = set()
    dupes = []
    for x in a:
        if x in seen:
            dupes.append(x)
        else:
            seen.add(x)
    return dupes

assessors = ['Both', 'Anne Marte', 'Jeanett']
assessments = dict()
assessments['all'] = list()

for assessor in assessors:
    assessments[assessor] = [word.split()[0].lower() for word in list(who[assessor]) if isinstance(word, str)]
    assessments['all'] += assessments[assessor]

print('checking uniqueness:')
for assessor in assessors:
    print('Duplicates for ', assessor, ':', duplicates(assessments[assessor]))

print('Duplicates for all:', duplicates(assessments['all']))

print('checking completeness:')
excels = [fn.split('/')[1][:-5] for fn in glob.glob(dir_path + '/*.xlsx') if fn.split('/')[1] != 'who did what.xlsx']

print('in who did what.xlsx but not in excel file:', set(assessments['all']) - set(excels))
print('in excel file, but not in who did what.xlsx:', set(excels) - set(assessments['all']))
