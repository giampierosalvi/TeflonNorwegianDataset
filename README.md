# TeflonNorwegianDataset
Scripts for the Norwegian version of the Teflon database. Note: this repository does not contain the data, but only scripts to convert the annotations into a csv file.

## Instructions
Install dependencies:
```
pip install openpyxl pandas playsound
```
Download the annotations zip file and unzip it and fix some filenames:
```
unzip 2023-03-02.zip
unzip All\ words.zip
unzip assessments.zip
rm -rf 023-03-02.zip All\ words.zip assessments.zip __MACOSX
mv All\ words speech
mv assessments/hund\ \(1\).xlsx assessments/hund.xlsx
mv assessments/grå.xlsx assessments/grå.xlsx
mv assessments/blå.xlsx assessments/blå.xlsx
mv assessments/tårn.xlsx assessments/tårn.xlsx
```
Check the `who did what.xlsx` file:
```
python3 check_who_did_what.py assessments
```
Then run
```
python3 fix_excel.py assessments
```
to fix the sheet names according to the assessor.

Then run
```
python3 assessments2csv.py assessments_fixed
```
this will generate the `assessments_fixed.csv` files with one row per audio file, and the following fields:
```
File name,Score,Prosody,Noise/Disruption,Pre-speech noise,Repetition,Word,Pronunciation,pronScores,Assessor
```
Both the 'Pronunciation' and 'pronScores' columns contain space separated items to cope with the fact that pronunciations can be of different lengths.

You can verify the data with:
```
python3 verify.py assessments.csv speech
```

You can play the examples and show the assessments with:
```
python3 play_examples.py assessments_fixed.csv speech
```
