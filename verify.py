import glob
import sys
import re
import os
import pandas as pd

csv_file = sys.argv[1]
audio_path = sys.argv[2]

df = pd.read_csv(csv_file)
print('Checking that every audio file from', csv_file, 'exists...')
for index, row in df.iterrows():
    # check if audio file exists
    audio_file_name = audio_path + '/' + row['File name']
    if not os.path.isfile(audio_file_name):
        print('File not found:', audio_file_name)
        alternatives = glob.glob(audio_file_name[:-4] + '*.wav')
        if len(alternatives):
            print('Found alternatives: ', alternatives)
print('Listing files in', audio_path, 'without an assessment:')
filenames = set(df['File name'])
for filename in glob.glob(audio_path + '/*.wav'):
    fn = filename.split('/')[1]
    if fn not in filenames:
        print(fn)

