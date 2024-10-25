# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Analysis of the dataset
# ## Giampiero Salvi
#

import numpy as np
import pandas as pd

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
data['gender'] = data['Gender'].map(gender_map)
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

# Fix language
language_map = dict()
for language in data['First Language'].unique():
    language_map[language] = language
language_map['Finnisjh'] = 'Finnish'
language_map['Ukraine'] = 'Ukrainian'
language_map['Engelsk',] = 'English'
language_map['Ukranian'] = 'Ukrainian'
language_map['English British'] = 'English'
language_map['Norsk'] = 'Norwegian'
data['First Language'] = data['First Language'].map(language_map)

data

uttdata = pd.read_csv('~/corpora/teflon_no/assessments.csv')

uttdata

uttdata['Speaker ID'] = [fn.split('_')[0] for fn in uttdata['File name']]

uttdata

scorehist = uttdata.groupby(['Speaker ID', 'Score']).size()

scoremeans = uttdata.groupby(['Speaker ID'])['Score'].mean()
scoremeans

scorehist.to_frame(name='count') #.pivot(index = 'Speaker ID', columns='Score', values='count')

uttdata.pivot(index='Speaker ID', columns='Score')


