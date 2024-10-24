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



column_names = ['submission_id', 'time', 'agreement', 'speaker_id', 'age', 'current_grade', 'gender', 'birth_country', 'immigration_age', 'exposed_to_dialect_oslo', 'exposed_to_dialect_other', 'exposed_to_dialect_region', 'first_language', 'second_language', 'third_language', 'fourth_language', 'first_language_level', 'second_language_level', 'third_language_level', 'fourth_language_level', 'used_languages', 'first_language_use', 'second_language_use', 'third_language_use', 'fourth_language_use', 'answer_time_ms']
data = pd.read_excel('data/Participant_Information_anonymised.xlsx', names=column_names)

data['age'].unique()

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
data['age'] = data['age'].map(age_map)

# fix genders
gender_map = {
    'Male': 'M',
    'Female': 'F',
    'Gutt': 'M',
    'Jente': 'F'
}
data['gender'] = data['gender'].map(gender_map)
#data['gender'].unique()

data['birth_country'].unique()

# Fix country
country_map = dict()
for country in data['birth_country'].unique():
    country_map[country] = country
country_map['Norge'] = 'Norway'
country_map['Norsk'] = 'Norway'
country_map['Ukrania'] = 'Ukraine'
country_map['Norway, but lived in the UK from age 3-10'] = 'Norway'
data['birth_country']= data['birth_country'].map(country_map)

data['first_language'].unique()

# Fix language
language_map = dict()
for language in data['first_language'].unique():
    language_map[language] = language
language_map['Finnisjh'] = 'Finnish'
language_map['Ukraine'] = 'Ukrainian'
language_map['Engelsk',] = 'English'
language_map['Ukranian'] = 'Ukrainian'
language_map['English British'] = 'English'
language_map['Norsk'] = 'Norwegian'
data['first_language'] = data['first_language'].map(language_map)

data


