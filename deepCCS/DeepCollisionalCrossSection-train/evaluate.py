from __future__ import print_function
#load_ext autoreload
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy, pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, os, json
from models import BiRNN_new, mlp, logreg
from data_util import get_data_set, one_hot_dataset, scale, unscale, int_dataset
from palettes import godsnot_64, zeileis_26
from bidirectional_lstm import predict
from data_util import Delta_t95, RMSE, Delta_tr95
from predict_util import get_color, plot_att
from predict_util import encode_data, get_tf_dataset
import matplotlib as mpl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sns.set(rc={"axes.facecolor":"#e6e6e6",
            "axes.grid":True,
               })

model_dir = './out/Tests_1/'
file_to_predict = 'data_final/Tests/2_test.pkl'
datasetname='4_batch'

with open(model_dir+'model_params.json') as f:
    model_params = json.load(f)

if file_to_predict != None:
    model_params['test_file'] = file_to_predict


model = BiRNN_new
print('Model: ', model)
figure_dir = model_params['model_dir'] + '/figures/'
lab_name = model_params['lab_name']
timesteps = model_params['timesteps']

with open('data_final/enc.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

c_dict = {}
for c,i in enumerate(label_encoder.classes_):
    c_dict[i]=godsnot_64[c]

#endregion
print('加载成功！')

#region part two 
test_data = pd.read_pickle(model_params['test_file'])
print('Using %s' % (model_params['test_file']))
test_sequences = test_data['Modified_sequence'].values


data = test_data
org_columns = data.columns

replaced_charge = False
try:
    data[model_params['lab_name']]
except:
    data[model_params['lab_name']]=np.zeros(len(data))
    replaced_charge = True

one_dat, lab, meta_data, test_size = encode_data(data, model_params)

#build iterator on testdata
dataset_test = get_tf_dataset(one_dat, lab, meta_data, data, model_params)
iter_test = dataset_test.make_initializable_iterator()
next_element_test = iter_test.get_next()

#endregion
print('读取成功！')


#region build graph 
# tf.reset_default_graph()
if model_params['simple']:
    X = tf.placeholder("float", [None, model_params['num_input']])
else:
    X = tf.placeholder("float", [None, model_params['timesteps']])
if model_params['num_classes'] == 1:
    Y = tf.placeholder("float", [None, 1])
else:
    Y = tf.placeholder("int64", [None, 1])
if model_params['num_tasks'] != -1:
    T = tf.placeholder("int32", [None])
else:
    T = None
C = tf.placeholder("float", [None, meta_data.shape[1]])
L = tf.placeholder("int32", [None])
dropout = tf.placeholder("float", ())

if model_params['num_tasks'] == -1:
    prediction, logits, weights, biases, attention, cert = model(X, C, L, model_params['num_layers'],
                                                                 model_params['num_hidden'], meta_data,
                                                                 model_params['num_classes'],
                                                                 model_params['timesteps'], keep_prob=dropout,
                                                                 uncertainty=model_params['use_uncertainty'],
                                                                 is_train=True)
else:
    prediction, logits, weights, biases, attention, cert = model(X, C, L, model_params['num_tasks'],
                                                                 model_params['num_layers'], model_params['num_hidden'],
                                                                 meta_data,
                                                                 model_params['num_classes'],
                                                                 model_params['timesteps'], keep_prob=dropout,
                                                                 uncertainty=model_params['use_uncertainty'],
                                                                 is_train=True)

if model_params['num_classes'] == 1:
    if model_params['num_tasks'] == -1:
        loss_op = tf.losses.mean_squared_error(predictions=prediction, labels=Y)
    else:  # multitask regression.

        pp = tf.reshape(tf.stack(prediction, axis=1), [-1, model_params['num_tasks']])
        ppp = tf.reshape(tf.reduce_sum(pp * tf.one_hot(T, model_params['num_tasks']), axis=1), [-1, 1])
        loss_op = tf.losses.mean_squared_error(predictions=ppp, labels=Y)
else:
    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(Y, [-1]), logits=prediction)
    loss_op = tf.reduce_mean(loss_op)
    prediction = tf.nn.softmax(prediction)

# Initialize the variables (i.e. assign their default value)
saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
# init model
init = [tf.global_variables_initializer(), iter_test.initializer]
# Start training
sess = tf.Session()
#endregion
print('tf成功！')


#region predictions 
sess.run(init)
model_file = tf.train.latest_checkpoint(model_params['model_dir'])
if model_file:
    ind1 = model_file.index('model')
    resume_itr = int(model_file[ind1+5:])
    print("Restoring model weights from " + model_file)
    saver.restore(sess, model_file)
else:
    print('no model found!')


n_preds = 1
for i in range(n_preds):
    label, preds, last, seq, charge, loss, att, unc, task  = predict(sess, X, Y, C, L, T, test_size, model_params, next_element_test, loss_op, prediction, logits, attention, meta_data, dropout, cert, dropout_rate = model_params['dropout_keep_prob'])
    #label, preds, last, seq, charge, loss, att, unc, task  = predict(sess, X, Y, C, L, T, test_size, model_params, next_element_test, loss_op, prediction, logits, attention, meta_data, dropout, cert, dropout_rate = 1.0)
    if model_params['num_classes'] != 1:
        preds = np.argmax(preds,axis=1).reshape(-1,1)
    data['label Prediction ' + str(i)] = preds[:,0]
    sess.run(iter_test.initializer)

#endregion
print('预测成功！')


#region 
df = data
mpl.rcParams.update(mpl.rcParamsDefault)
inline_rc = dict(mpl.rcParams)
sns.set(rc={"axes.facecolor": "#ffffff",
            "axes.grid": False,
            })
sns.set_style('ticks')
sns.despine()

# df = pd.read_hdf(figure_dir+'data.h5')

if replaced_charge:
    data['Charge'] = data['Charge Prediction']

print(data[['Modified_sequence',model_params['lab_name'],model_params['lab_name']+' Prediction 0']].head())

set(data['Charge'].values)

if model_params['lab_name'] != 'Charge':
    data['label'] = data['CCS']


data[['Modified_sequence','Charge','label Prediction 0']].to_csv(figure_dir + 'prediction_'+model_params['lab_name']+'_'+datasetname+'.csv')
print('saved to', figure_dir + 'prediction_'+model_params['lab_name']+'_'+datasetname+'.csv')
#endregion
print('保存成功！')

print(df.columns)

#region Plots
df['rel'] = (df[lab_name] / df[lab_name+' Prediction 0'] ) * 100 - 100
df['abs'] = np.abs(df[lab_name] - df[lab_name+' Prediction 0'])

rel = df['rel'].values
rel_abs = np.abs(df['rel'].values)
abs = df['abs'].values
print(np.median(rel_abs))


median_rel = np.median(rel)


absolute_deviation = np.abs(rel - median_rel)

mad = np.median(absolute_deviation)
print("Absolute Median Deviation (MAD):", mad)


ax = sns.distplot(rel, norm_hist=False, kde = False, bins=50)    #, bins=200
ax.set(xlabel='deviation (%)', ylabel='Counts')
ax.set(xlim = [-10, 10])
plt.tight_layout()
sns.despine()
plt.title('Peptide CCS Prediction Deviation')
plt.savefig(figure_dir + '/rel_error2.svg', dpi=300)
plt.savefig(figure_dir + '/rel_error2.pdf', dpi=300, format='pdf')  # 

ppred = df[lab_name+' Prediction 0']
llabel = df[lab_name]
#endregion
print('plots成功！')
plt.close()

#region pearson
ax = sns.regplot(x=df[lab_name], y=df[lab_name+' Prediction 0'],scatter_kws={'s':0.02})#, scatter_kws={'color' : ccs}
pearson = scipy.stats.pearsonr(df[lab_name+' Prediction 0'], df[lab_name])
print('Pearson', pearson[0])
ax.set(xlabel='observed CCS', ylabel='predicted CCS')

plt.text(700,450,'Pearson: {:.4f}'.format(pearson[0]))

plt.title('Pearson Correlation')
sns.despine()
plt.savefig(figure_dir + '/pearson2.png', dpi=300)

#endregion
print('pearson成功！')




















