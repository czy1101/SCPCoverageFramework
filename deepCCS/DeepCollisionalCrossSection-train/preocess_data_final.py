import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re, os, csv
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
#region methond
def remove10(data):
    # remove upper/lower 10 percent
    print(data.shape)
    low = np.percentile(data['Sum_intensity'], 10)
    high = np.percentile(data['Sum_intensity'], 90)
    print(low, high)
    d = data
    d = d[d['Sum_intensity'] > low]
    # d = d[d['Sum_intensity'] < high]
    print(d.shape)
    # d6 = d6.groupby(['Modified_sequence', 'Charge'], group_keys=False).apply(lambda x: x.loc[x.Intensity.idxmax()])
    return d

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def split(data, name, s, label_encoder_path='data_final/enc.pickle', ids=None, calc_minval=True):
    ensure_dir(name)
    np.random.seed(s)
    with open(label_encoder_path, 'rb') as handle:
        label_encoder = pickle.load(handle)
    data['encseq'] = data['Modified_sequence'].apply(lambda x: label_encoder.transform(list(x)))
    if calc_minval:
        data['minval'] = np.min(data['label'])
        data['maxval'] = np.max(data['label'])
    else:
        data['minval'] = 275.440277
        data['maxval'] = 1112.030762

    if ids == None:
        # a = np.random.uniform(0.0,1.0,len(data)) > 0.98
        data['test'] = data['PT']
        print('Using proteome tools testsset')

    else:
        print('using predefined testset')
        data['test'] = ~data['Modified_sequence'].isin(ids)

    data['task'] = 0
    print('Name: ', name, 'Seed: ', s, 'Len test: ', len(data[data['test']]), 'Len set test: ',
          len(set(data[data['test']])), 'Len not test: ', len(data[~data['test']]), 'Len set not test: ',
          len(set(data[~data['test']])))
    data[~data['test']].to_pickle(name + str(s) + '_train.pkl')
    data[data['test']].to_pickle(name + str(s) + '_test.pkl')
    return data


def kill_new_seqs(df):
    df['label'] = np.zeros(len(df))
    print(len(df), 'total seqs')
    seqs = df['Modified_sequence'].values.tolist()
    flat_list = [item for sublist in seqs for item in sublist]
    s = set(flat_list)
    with open('data_final/enc_list.csv') as f:
        reader = csv.reader(f)
        enc_list = list(reader)[0]
        print(enc_list)
    new_acids = list(set(s) - set(enc_list))
    if len(new_acids) != 0:
        df_red = df[~df['Modified_sequence'].str.contains('|'.join(new_acids))]
    else:
        df_red = df
    print(len(df_red), 'reduced seqs', new_acids)
    return df_red
#endregion




#data_path = 'data/combined.csv'
data_path = '../data_csv/combined_ok.csv'

outpath = 'data_final/Tests/'
new_characters = ['U']

data6 = pd.read_csv(data_path)
data6 = data6.rename(index=str, columns={"Modified sequence": "Modified_sequence"})
data6['Modified_sequence'] = data6['Modified_sequence'].str.replace('_','')

data = data6
dat = data['Modified_sequence']
dat = [list(d) for d in dat]

# process data into one hot encoding
flat_list = ['_'] + [item for sublist in dat for item in sublist]

#  
flat_list.extend(new_characters)

# define example
values = np.array(flat_list)

label_encoder = LabelEncoder()
label_encoder.fit(values)

print(label_encoder.classes_, len(label_encoder.classes_))
with open('data_final/enc.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle)

with open('data_final/enc_list.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(list(label_encoder.classes_))

data6 = data6[~data6['Intensity'].isnull()]
data6 = data6[~data6['CCS'].isnull()]
#data6['Modified_sequence'] = data6['Modified_sequence'].str.replace('_','')

# for multiple with same sequence and charge take the one with highest intensity
#d6 = data6.groupby(['Modified_sequence', 'Charge'], group_keys=False).apply(lambda x: x.loc[x.Intensity.idxmax()])
d6 = data6

d6['label']=d6['CCS'].values.tolist()
dd = split(d6, outpath, 1)

trainseqs = dd[~dd['test']]['Modified_sequence'].values.tolist()
ddd = dd[dd['test']]
ddd[ddd['Modified_sequence'].isin(trainseqs)].shape

dd = split(d6, outpath, 2, ids=trainseqs)
trainseqs = dd[~dd['test']]['Modified_sequence'].values.tolist()
ddd = dd[dd['test']]
ddd[ddd['Modified_sequence'].isin(trainseqs)].shape
print(len(ddd))

