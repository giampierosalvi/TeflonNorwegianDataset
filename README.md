# TeflonNorwegianDataset
Scripts for the Norwegian version of the Teflon database. Note: this repository does not contain the data, but only scripts to convert the annotations into a csv file.

## Instructions
Install dependencies:
```
pip install openpyxl
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
annotations2csv.py annotations
```
