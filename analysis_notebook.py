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

print(data.columns)

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

scorehist.boxplot()
plt.show()

# merge speaker information with per-speaker assessment information
datafull = pd.merge(data, scorehist, on='Speaker ID')
datafull.columns


# calculate average score
def countedAverage(counts):
    values = np.arange(len(counts))+1
    N = counts.sum()
    return np.inner(values, counts)/N
datafull['Average Score'] = datafull[['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']].apply(countedAverage, axis = 1)

# assign each speaker to a quartile based on average score
#datafull['Score Range'] = pd.qcut(datafull['Average Score'], q=4, labels=['low', 'low_med', 'high_med', 'high'])
#datafull['Score Range'] = pd.qcut(datafull['Average Score'], q=3, labels=['low', 'med', 'high'])
datafull['Score Range'] = pd.qcut(datafull['Average Score'], q=2, labels=['low', 'high'])

# group age ranges
datafull['Age Range'] = pd.qcut(datafull['Age'], q=2, labels=['4-9', '10-12'])
#datafull = datafull.set_index('Speaker ID', drop=False)

N = datafull.shape[0] # total number of speakers
N_test = 8            # desired number of speakers in the test set
# define speaker classes and count number of speakers per class
spk_class = dict()
count = dict()
lang_class = { # L1: mother tongue Norwegian, CL2: common foreign, UL2: uncommon foreign
    'Norwegian': 'Nor',
    'Ukrainian': 'Ukr',
    'English': 'Eng',
    'Russian': 'Rus',
    'Finnish': 'UL2',
    'Estonian': 'UL2',
    'Persian': 'UL2',
    'Urdu': 'UL2',
    'Albanian': 'UL2',
    'Vietnamese': 'UL2',
    'Dutch': 'UL2',
    'Mandarin': 'UL2',
    'French': 'UL2'
}
trainset = dict()
for index, row in datafull.iterrows():
    # first move all speakers with very uncommon language background to the training set
    if lang_class[row['First Language']] == 'UL2':
        trainset[row['Speaker ID']] = True
        continue
    #lang_bg = 'L1' if row['First Language']=='Norwegian' else 'L2'
    cl = '_'.join((lang_class[row['First Language']], row['Gender'], row['Age Range'], row['Score Range']))
    spk_class[row['Speaker ID']] = cl
    if cl in count.keys():
        count[cl] += 1
    else:
        count[cl] = 1
for k, v in count.items():
    print("{:<20} {:<10}".format(k, v))
# Generate optimum (non integer) speaker distribution
optimal = dict()
diff = dict()
for cl in count.keys():
    optimal[cl] = count[cl]*N_test/N
    diff[cl] = count[cl]-optimal[cl]
# iteratively remove speakers from the test set until desired #speakers is reached
random.seed(10)
while sum(count.values()) > N_test:
    top = max(diff, key=diff.get) # select the class with largest difference between #speakers and optimum
    # select all speakers that correspond to this class but are not already in the training set
    spk_set = [k for k, v in spk_class.items() if v == top and not k in trainset.keys()]
    # select one such speaker at random
    sp = random.choice(spk_set)
    trainset[sp] = True
    count[top] = count[top]-1
    diff[top] = count[top]-optimal[top]
trainids = list(trainset.keys())
trainids.sort()
testids = list(set(datafull['Speaker ID']) - set(trainset.keys()))
testids.sort()
print('Training set: ', trainids)
print('Test set: ', testids)
for k, v in count.items():
    print("{:<20} {:<10}".format(k, v))

with open('training_set.lst', 'w') as outfile:
    outfile.write('\n'.join(trainids))
    outfile.write('\n')
with open('test_set.lst', 'w') as outfile:
    outfile.write('\n'.join(testids))
    outfile.write('\n')

diff

# add infromation about training and test set to uttdata
id2age = dict()
id2fl = dict()
for index, row in datafull.iterrows():
    id2age[row['Speaker ID']] = row['Age']
    id2fl[row['Speaker ID']] = row['First Language']
partition = list()
age = list()
first_lang = list()
for index, row in uttdata.iterrows():
    partition.append('train' if row['Speaker ID'] in trainids else 'test')
    age.append(id2age[row['Speaker ID']])
    first_lang.append(id2fl[row['Speaker ID']])
uttdata['Partition'] = partition
uttdata['Age'] = age
uttdata['First Language'] = first_lang

import seaborn as sns
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=uttdata, x='Score', hue='Partition', hue_order=['train', 'test'], ax=axs[0])
axs[0].set_title('Score Distribution (counts)')
prop_uttdata = (uttdata['Score'].groupby(uttdata['Partition']).value_counts(normalize=True).rename('percentage').reset_index())
#sns.countplot(data=uttdata, x='Score', hue='Partition', stat='percent', ax=axs[1])
sns.barplot(data=prop_uttdata, x='Score', y='percentage', hue='Partition', hue_order=['train', 'test'], ax=axs[1])
axs[1].set_title('Score Distribution (%)')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=uttdata, x='Age', hue='Partition', hue_order=['train', 'test'], ax=axs[0])
axs[0].set_title('Age Distribution (counts)')
prop_uttdata = (uttdata['Age'].groupby(uttdata['Partition']).value_counts(normalize=True).rename('percentage').reset_index())
sns.barplot(data=prop_uttdata, x='Age', y='percentage', hue='Partition', hue_order=['train', 'test'], ax=axs[1])
axs[1].set_title('Age Distribution (%)')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
languages = ['Norwegian', 'English', 'Ukrainian', 'Russian', 'Finnish',
       'Persian', 'Urdu', 'Albanian', 'Vietnamese',
       'Dutch', 'Mandarin', 'French', 'Estonian']
sns.countplot(data=uttdata, x='First Language', order=languages, hue='Partition', hue_order=['train', 'test'], ax=axs[0])
axs[0].tick_params(axis='x', rotation=90)
axs[0].set_title('First Language Distribution (counts)')
prop_uttdata = (uttdata['First Language'].groupby(uttdata['Partition']).value_counts(normalize=True).rename('percentage').reset_index())
sns.barplot(data=prop_uttdata, x='First Language', order=languages, y='percentage', hue='Partition', hue_order=['train', 'test'], ax=axs[1])
axs[1].tick_params(axis='x', rotation=90)
axs[1].set_title('First Language Distribution (%)')
plt.show()

# assign random file name
names = ["{:04}.wav".format(x) for x in np.arange(10000)]
random.seed(0)
random.shuffle(names)
nameMap = dict()
idx = 0
for original in uttdata['File name'].unique():
    nameMap[original] = names[idx]
    idx +=1
fileNameSeries = list()
for index, row in uttdata.iterrows():
    fileNameSeries.append(nameMap[row['File name']])
uttdata['Challenge File Name'] = fileNameSeries

# get all scores for each file
scores = dict()
for index, row in uttdata.iterrows():
    if row['File name'] not in scores.keys():
        scores[row['File name']] = list()
    scores[row['File name']].append(row['Score'])
# convert scores lists into single scores
to_remove = list()
for fn in scores.keys():
    if 0 in scores[fn]:
        to_remove.append(fn)
    scores[fn] = int(np.rint(np.array(scores[fn]).mean()))
for fn in to_remove:
    del scores[fn]

partitionMap = dict()
for id in trainids:
    partitionMap[id] = 'train'
for id in testids:
    partitionMap[id] = 'test'

# create datasets
import os
import shutil
for newpath in ['TeflonChallenge/train', 'TeflonChallenge/test']:
    if not os.path.exists(newpath):
        os.makedirs(newpath)
#train_df = pd.DataFrame(columns=['File Name', 'Word', 'Score'])
#public_test_df = pd.DataFrame(columns=['File Name', 'Word'])
#private_test_df = pd.DataFrame(columns=['File Name', 'Word', 'Score'])
train_df = list()
public_test_df = list()
private_test_df = list()
for fn in scores.keys():
    outfn = nameMap[fn]
    spkrid, word = fn[:-4].split('_')
    partition = partitionMap[spkrid]
    outdir = 'TeflonChallenge/' + partition
    shutil.copy('speech/'+fn, outdir+'/'+outfn)
    if partition == 'train':
        train_df.append({'File Name': outfn, 'Word': word, 'Score': scores[fn]})
    else:
        public_test_df.append({'File Name': outfn, 'Word': word})
        private_test_df.append({'File Name': outfn, 'Word': word, 'Score': scores[fn]})
pd.DataFrame(train_df).to_csv('TeflonChallenge/train.csv', index=False)
pd.DataFrame(public_test_df).to_csv('TeflonChallenge/test.csv', index=False)
pd.DataFrame(private_test_df).to_csv('private_test.csv', index=False)



# +
# put data on https://zenodo.org/
