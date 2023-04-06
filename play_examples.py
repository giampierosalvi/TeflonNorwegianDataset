from playsound import playsound
import sys
import pandas as pd

csv_file = sys.argv[1]
audio_path = sys.argv[2]

df = pd.read_csv(csv_file)
for index, row in df.iterrows():
    print(row)
    playsound(audio_path + '/' + row['File name'])
