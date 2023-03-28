# TeflonNorwegianDataset
Scripts for the Norwegian version of the Teflon database. Note: this repository does not contain the data, but only scripts to convert the annotations into a csv file.

## Instructions
Install dependencies:
```
pip install openpyxl pandas
```
Download the annotations zip file and unzip it and fix some filenames:
```
unzip 2023-03-02.zip
unzip All\ words.zip
unzip assessments.zip
rm -rf 023-03-02.zip All\ words.zip assessments.zip __MACOSX
mv All\ words speech
mv assessments/hund\ \(1\).xlsx assessments/hund.xlsx
```
Then run
```
python3 assessments2csv.py assessments
```
this will generate the `assessments.csv` files with one row per audio file, and the following fields:
```
File name,Score,Prosody,Noise/Disruption,Pre-speech noise,Repetition,Word,Pronunciation,pronScores
```
Both the 'Pronunciation' and 'pronScores' columns contain space separated items to cope with the fact that pronunciations can be of different lengths.

You can verify the data with:
```
python3 verify.py assessments.csv speech
```
