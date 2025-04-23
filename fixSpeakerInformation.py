# This script fixes the speaker information stored in
# data/Participant_Information_anonymised.xlsx
# it also adds statistics of the scores for each speaker computed from data/assessments.csv
#
# Finally is saves the resulting data frame in data/speaker_info.csv
# It only needs to be run once.
#
# (C) 2025 Giampiero Salvi <giampiero.salvi@ntnu.no>
#

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

column_names = ['Submission ID', 'Time', 'Agreement', 'Speaker ID', 'Age', 'Current Grade', 'Gender', 'Birth Country', 'Immigration Age', 'Exposed To Dialect Oslo', 'Exposed To Dialect Other', 'Exposed To Dialect Region', 'First Language', 'Second Language', 'Third Language', 'Fourth Language', 'First Language Level', 'Second Language Level', 'Third Language Level', 'Fourth Language Level', 'Used Languages', 'First Language Use', 'Second Language Use', 'Third Language Use', 'Fourth Language Use', 'Answer Time ms']
data = pd.read_excel('data/Participant_Information_anonymised.xlsx', names=column_names)

# fix ages
age_map = {
    '7 år': 7,
    '5,5 år': 5,
    '9 År': 9,
    '9 år (straks ti)': 9,
    '8 år': 8,
    '5 år (fyller 6 år 14. mai)': 5
}
for age in range(4,13):
    age_map[str(age)] = age
data['Age'] = data['Age'].map(age_map)

# fix genders
gender_map = {
    'Male': 'M',
    'Female': 'F',
    'Gutt': 'M',
    'Jente': 'F'
}
data['Gender'] = data['Gender'].map(gender_map)
#data['gender'].unique()

# Fix country
country_map = dict()
for country in data['Birth Country'].unique():
    country_map[country] = country
country_map['Norge'] = 'Norway'
country_map['Norsk'] = 'Norway'
country_map['Ukrania'] = 'Ukraine'
country_map['Norway, but lived in the UK from age 3-10'] = 'Norway'
data['Birth Country']= data['Birth Country'].map(country_map)

# Fix Speaker ID
speaker_id_map = dict()
for id in data['Speaker ID']:
    speaker_id_map[id] = id
speaker_id_map['d18/p03'] = 'd18'
speaker_id_map['d19/p04'] = 'd19'
data['Speaker ID'] = data['Speaker ID'].map(speaker_id_map)

# Fix language
language_map = dict()
for language in data['First Language'].unique():
    language_map[language] = language
language_map['Finnisjh'] = 'Finnish'
language_map['Ukraine'] = 'Ukrainian'
language_map['Engelsk',] = 'English'
language_map['Ukranian'] = 'Ukrainian'
language_map['English British'] = 'English'
language_map['Engelsk'] = 'English'
language_map['Norsk'] = 'Norwegian'
data['First Language'] = data['First Language'].map(language_map)

uttdata = pd.read_csv('data/assessments.csv')

uttdata['Speaker ID'] = [fn.split('_')[0] for fn in uttdata['File name']]

# count occurrences of each score for each speaker
scorehist = uttdata.groupby(['Speaker ID', 'Score']).size()
scorehist = scorehist.to_frame(name='count').reset_index()
# convert to table format and fill missing data with zeros
scorehist = scorehist.pivot(index = 'Speaker ID', columns='Score', values='count').fillna(0).astype('int')
column_names = list()
for score in range(6):
    column_names.append('Score '+str(score))
# rename columns
scorehist.columns = pd.Index(column_names)
# convert index to column
scorehist = scorehist.reset_index()

# scorehist.boxplot()
# plt.show()

# merge speaker information with per-speaker assessment information
datafull = pd.merge(data, scorehist, on='Speaker ID')

datafull.to_csv('data/speaker_info.csv')
