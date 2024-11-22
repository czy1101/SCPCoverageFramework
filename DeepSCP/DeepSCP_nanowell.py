# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : wangbing
 
# @Desc     : DeepSCP: utilizing deep learning to boost single-cell proteome coverage


import numpy as np
import pandas as pd
import scipy as sp
import lightgbm as lgb
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

from copy import deepcopy
from time import time
from joblib import Parallel, delayed
from scipy.stats.mstats import gmean
from scipy.stats import pearsonr

from bayes_opt import BayesianOptimization
from triqler.qvality import getQvaluesFromScores

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import warnings
import math

from mamba_ssm import Mamba2
from mamba_ssm import Mamba
warnings.filterwarnings('ignore')
#plt.rcParams['font.sans-serif'] = 'Arial'

#region method
def showcols(df):
    cols = df.columns.tolist()
    cols = cols + (10 - len(cols) % 10) % 10 * ['None']
    cols = np.array(cols).reshape(-1, 10)
    return pd.DataFrame(data=cols, columns=range(1, 11))


def Prob2PEP(y, y_prob):
    label = np.array(deepcopy(y))
    score = np.array(deepcopy(y_prob))
    srt_idx = np.argsort(-score)
    label_srt = label[srt_idx]
    score_srt = score[srt_idx]
    targets = score_srt[label_srt == 1]
    decoys = score_srt[label_srt != 1]
    _, pep = getQvaluesFromScores(targets, decoys, includeDecoys=True)
    return pep[np.argsort(srt_idx)]


def Score2Qval(y0, y_score):
    y = np.array(deepcopy(y0)).flatten()
    y_score = np.array(y_score).flatten()
    y[y != 1] = 0
    srt_idx = np.argsort(-y_score)
    y_srt = y[srt_idx]
    cum_targets = y_srt.cumsum()
    cum_decoys = np.abs(y_srt - 1).cumsum()
    FDR = np.divide(cum_decoys, cum_targets)
    qvalue = np.zeros(len(FDR))
    qvalue[-1] = FDR[-1]
    qvalue[0] = FDR[0]
    for i in range(len(FDR) - 2, 0, -1):
        qvalue[i] = min(FDR[i], qvalue[i + 1])
    qvalue[qvalue > 1] = 1
    return qvalue[np.argsort(srt_idx)]


def GroupProteinPEP2Qval(data, file_column, protein_column, target_column, pep_column):
    data['Protein_label'] = data[protein_column] + '_' + data[target_column].map(str)
    df = []
    for i, j in data.groupby(file_column):
        df_pro = j[['Protein_label', target_column, pep_column]].sort_values(pep_column).drop_duplicates(
            subset='Protein_label', keep='first')
        df_pro['protein_qvalue'] = Score2Qval(df_pro[target_column].values, -df_pro[pep_column].values)
        df.append(pd.merge(j,
                           df_pro[['Protein_label', 'protein_qvalue']],
                           on='Protein_label',
                           how='left'))
    return pd.concat(df, axis=0).drop('Protein_label', axis=1)


class SampleRT:
    def __init__(self, r=3):
        self.r = r

    def fit_tranform(self, data, file_column, peptide_column, RT_column, score_column, target_column):
        self.file_column = file_column  # 'experiment'
        self.peptide_column = peptide_column  # 'modified sequence'
        self.RT_column = RT_column  # 'Retention time'
        self.score_column = score_column  # 'score'
        self.target_column = target_column  # 'label'
        
        data['File_Pep'] = data[self.file_column] + '+' + data[self.peptide_column]
        dftagraw = data[data[self.target_column] == 1]
        dfrevraw = data[data[self.target_column] != 1]


        dftag1 = self.makePeptied_File(self.Repeat3(dftagraw))  
        dfrev1 = self.makePeptied_File(self.Repeat3(dfrevraw))
        dftag2 = self.makeRTfeature(dftag1)  
        dfrev2 = self.makeRTfeature(dfrev1)

        reg = ElasticNet()

        pred_tag_tag = pd.DataFrame(data=np.zeros_like(dftag1.values), columns=dftag1.columns, index=dftag1.index)
        pred_rev_tag = pd.DataFrame(data=np.zeros_like(dfrev1.values), columns=dfrev1.columns, index=dfrev1.index)
        pred_rev_rev = pd.DataFrame(data=np.zeros_like(dfrev1.values), columns=dfrev1.columns, index=dfrev1.index)
        pred_tag_rev = pd.DataFrame(data=np.zeros_like(dftag1.values), columns=dftag1.columns, index=dftag1.index)

        scores_tag_tag = []
        scores_rev_tag = []
        scores_tag_rev = []
        scores_rev_rev = []

        samples = dftag1.columns.tolist()
        for sample in samples:  
            y_tag = dftag1[~dftag1[sample].isna()][sample]
            X_tag = dftag2.loc[dftag2.index.isin(y_tag.index)]

            y_rev = dfrev1[~dfrev1[sample].isna()][sample]
            X_rev = dfrev2.loc[dfrev2.index.isin(y_rev.index)]

            reg_tag = reg
            reg_tag.fit(X_tag, y_tag)

            scores_rev_tag.append(reg_tag.score(X_rev, y_rev))
            scores_tag_tag.append(reg_tag.score(X_tag, y_tag))

            pred_rev_tag.loc[dfrev2.index.isin(y_rev.index), sample] = reg_tag.predict(X_rev)
            pred_tag_tag.loc[dftag2.index.isin(y_tag.index), sample] = reg_tag.predict(X_tag)

            reg_rev = reg
            reg_rev.fit(X_rev, y_rev)

            scores_rev_rev.append(reg_rev.score(X_rev, y_rev))
            scores_tag_rev.append(reg_rev.score(X_tag, y_tag))

            pred_rev_rev.loc[dfrev2.index.isin(y_rev.index), sample] = reg_rev.predict(X_rev)
            pred_tag_rev.loc[dftag2.index.isin(y_tag.index), sample] = reg_rev.predict(X_tag)

            pred_rev_tag[pred_rev_tag == 0.0] = np.nan
            pred_tag_tag[pred_tag_tag == 0.0] = np.nan
            pred_rev_rev[pred_rev_rev == 0.0] = np.nan
            pred_tag_rev[pred_tag_rev == 0.0] = np.nan

        self.cmp_scores = pd.DataFrame({'score': scores_tag_tag + scores_rev_tag + scores_tag_rev + scores_rev_rev,
                                        'type': ['RT(tag|tag)'] * len(scores_tag_tag) + ['RT(rev|tag)'] * len(
                                            scores_rev_tag) +
                                                ['RT(tag|rev)'] * len(scores_tag_rev) + ['RT(rev|rev)'] * len(
                                            scores_tag_rev)})

        pred_rev = pd.merge(self.makeRTpred(pred_rev_rev, 'RT(*|rev)'),
                            self.makeRTpred(pred_rev_tag, 'RT(*|tag)'), on='File_Pep')
        dfrevraw = pd.merge(dfrevraw, pred_rev, on='File_Pep', how='left')

        pred_tag = pd.merge(self.makeRTpred(pred_tag_rev, 'RT(*|rev)'),
                            self.makeRTpred(pred_tag_tag, 'RT(*|tag)'), on='File_Pep')
        dftagraw = pd.merge(dftagraw, pred_tag, on='File_Pep', how='left')

        df = pd.concat([dftagraw, dfrevraw], axis=0)
        df['DeltaRT'] = ((df[self.RT_column] - df['RT(*|rev)']).apply(abs) +
                         (df[self.RT_column] - df['RT(*|tag)']).apply(abs))
        df['DeltaRT'] = df['DeltaRT'].apply(lambda x: np.log2(x + 1))
        return df

    def Repeat3(self, data):
        df = pd.Series([i.split('+')[1] for i in data['File_Pep'].unique()]).value_counts()
        return data[data[self.peptide_column].isin(df[df >= self.r].index)]

    def makePeptied_File(self, data):
        data1 = data.sort_values(self.score_column, ascending=False).drop_duplicates(subset='File_Pep',
                                                                                     keep='first')[
            [self.file_column, self.peptide_column, self.RT_column]]
        temp1 = list(zip(data1.iloc[:, 0], data1.iloc[:, 1], data1.iloc[:, 2]))
        G = nx.Graph()
        G.add_weighted_edges_from(temp1)
        df0 = nx.to_pandas_adjacency(G)
        df = df0[df0.index.isin(data1[self.peptide_column].unique())][data1[self.file_column].unique()]
        df.index.name = self.peptide_column
        df[df == 0.0] = np.nan
        return df

    def makeRTfeature(self, data):
        df_median = data.median(1)
        df_gmean = data.apply(lambda x: gmean(x.values[~np.isnan(x.values)]), axis=1)
        df_mean = data.mean(1)
        df_std = data.std(1)
        df_cv = df_std / df_mean * 100
        df_skew = data.apply(lambda x: x.skew(), axis=1)
        df = pd.concat([df_median, df_gmean, df_mean, df_std, df_cv, df_skew], axis=1)
        df.columns = ['Median', 'Gmean', 'Mean', 'Std', 'CV', 'Skew']
        return df

    def makeRTpred(self, data, name):
        m, n = data.shape
        cols = np.array(data.columns)
        inds = np.array(data.index)
        df_index = np.tile(inds.reshape(-1, 1), n).flatten()
        df_columns = np.tile(cols.reshape(1, -1), m).flatten()
        values = data.values.flatten()
        return pd.DataFrame({'File_Pep': df_columns + '+' + df_index, name: values})


class MQ_SampleRT:
    def __init__(self, r=3, filter_PEP=False):
        self.r = r
        self.filter_PEP = filter_PEP

    def fit_tranform(self, data):
        if self.filter_PEP:
            data = data[data['PEP'] <= self.filter_PEP]
        df = pd.concat([self.get_tag(data), self.get_rev(data)], axis=0)
        sampleRT = SampleRT(r=self.r)
        dfdb = sampleRT.fit_tranform(df,
                                     file_column='Experiment',
                                     peptide_column='Modified sequence',
                                     RT_column='Retention time',
                                     score_column='Score',
                                     target_column='label')

        self.cmp_scores = sampleRT.cmp_scores
        dfdb['PEPRT'] = dfdb['DeltaRT'] * (1 + dfdb['PEP'])
        dfdb['ScoreRT'] = dfdb['Score'] / (1 + dfdb['DeltaRT'])
        return dfdb

    def get_tag(self, df):
        df = df[~df['Proteins'].isna()]
        df_tag = df[(~df['Proteins'].str.contains('CON_')) & (~df['Leading razor protein'].str.contains('REV__'))]
        df_tag['label'] = 1
        return df_tag

    def get_rev(self, df):
        df_rev = df[
            (df['Leading razor protein'].str.contains('REV__')) & (~df['Leading razor protein'].str.contains('CON__'))]
        df_rev['label'] = 0
        return df_rev


# region DeepSpec 

class DeepSpec:
    def __init__(self, model=None, seed=0, test_size=0.2, lr=1e-3, l2=0.0,
                 batch_size=256, epochs=1000, nepoch=50, patience=50,
                 device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
                 # device=torch.device("cpu")
                 ):
        self.test_size = test_size
        self.seed = seed
        self.batch_size = batch_size
        self.device = device
        self.patience = patience
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.nepoch = nepoch
        self.model = model

    def fit(self, bkmsms):
        print('+++++++++++++++++++++++++++Loading Trainset+++++++++++++++++++++')
        bkmsms['CMS'] = bkmsms['Charge'].map(str) + bkmsms['Modified sequence']
        # bkmsms['CMS'] = bkmsms['CMS'].apply(lambda x: x.replace('_(ac)M(ox)', 'B').replace(
        #     '_(ac)', 'J').replace('M(ox)', 'O').replace('_', ''))
        bkmsms['CMS'] = bkmsms['CMS'].apply(
            lambda x: x.replace('(Acetyl (Protein N-term))M(Oxidation (M))', 'B').replace(
                '(Acetyl (Protein N-term))', 'J').replace('(Oxidation (M))', 'O').replace('_', ''))

        bkmsms1 = self.selectBestmsms(bkmsms, s=100)[['CMS', 'Matches', 'Intensities']]

        x, y = IonCoding().fit_transfrom(bkmsms1)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.test_size, random_state=self.seed)
        y_true = np.array(y_val).reshape(y_val.shape[0], -1).tolist()
        train_db = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_db,
                                  batch_size=self.batch_size,
                                  num_workers=0,
                                  shuffle=True)

        val_db = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_db,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=False)

        if self.model is None:
            torch.manual_seed(self.seed)
            self.model = CNN_BiLSTM(self.batch_size, 30, 256, 2, 12, self.device)
        #  DataParallel
        # if torch.cuda.device_count() > 1:
        #     #model = nn.DataParallel(self.model).to(self.device)
        # else:
        #     model = self.model.to(self.device)
        model = self.model.to(self.device)

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2)

        val_losses = []
        val_cosines = []
        val_MSEs = []
        val_MAEs = []
        val_pearsons = []
        val_frobeniuss = []

        self.val_cosine_best = 0.0
        self.val_F_best = 500  
        self.val_loss = 100  
        counter = 0

        # region 
        if not os.path.exists('nanowell2.txt'):
            with open('nanowell2.txt', 'w') as f:
                pass  
        # endregion

        print('+++++++++++++++++++DeepSpec Training+++++++++++++++++++++', len(train_loader))
        for epoch in range(1, self.epochs + 1):
            for i, (x_batch, y_batch) in enumerate(train_loader):
                print(
                    f'current epoch: {epoch}, progress: {i}/{len(train_loader)} ({(i / len(train_loader)) * 100:.2f}%)')
                model.train()

                batch_x = x_batch.to(self.device)
                batch_y = y_batch.to(self.device)
                out = model(batch_x)

                loss = loss_func(out, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = 0
                y_valpreds = []
                for a, b in val_loader:
                    val_x = a.to(self.device)
                    val_y = b.to(self.device)
                    y_valpred = model(val_x)
                    y_valpreds.append(y_valpred)
                    val_loss += loss_func(y_valpred, val_y).item() / len(val_loader)

                val_losses.append(val_loss)
                y_valpreds = torch.cat([y_vp for y_vp in y_valpreds], dim=0)
                y_pred = np.array(y_valpreds.cpu()).reshape(y_val.shape[0], -1).tolist()

                # region 
            
                val_cosine = self.cosine_similarity(y_true, y_pred)  
                val_cosines.append(val_cosine)

                val_MSE = self.calculate_mse(y_true, y_pred) 
                val_MSEs.append(val_MSE)

                val_MAE = self.calculate_mae(y_true, y_pred)  
                val_MAEs.append(val_MAE)

                val_pearson = self.calculate_pearson(y_true, y_pred)
                val_pearsons.append(val_pearson)

                val_frobenius = self.frobenius_norm_of_difference(y_true, y_pred)
                val_frobeniuss.append(val_frobenius)
                # endregion

                if val_cosine.mean() >= self.val_cosine_best:
                    counter = 0
                    self.val_cosine_best = val_cosine.mean()
                    self.val_loss_best = val_loss
                    self.bestepoch = epoch
                    torch.save(model, 'DeepSpec_nanowell2.pkl')
                else:
                    counter += 1

                # region 
                with open('nanowell2.txt', 'a') as f:
                    # log_message = '[{}|{}] val_loss: {} | val_cosine: {}'.format(epoch, self.epochs, val_loss,val_cosine.mean())
                    log_message = '[{}|{}] val_loss: {} | val_cosine: {} | val_MSEs: {} | val_MAEs: {} | val_pearsons: {} | val_frobeniuss: {}'.format(
                        epoch,
                        self.epochs,
                        val_loss,
                        val_cosine.mean(),
                        val_MSE,  #  val_MSEs 
                        val_MAE,  #  val_MAEs 
                        val_pearson,  #  val_pearsons 
                        val_frobenius  #  val_frobeniuss 
                    )
                    f.write(log_message + '\n')
                # endregion

                if epoch % self.nepoch == 0 or epoch == self.epochs:
                    print(
                        '[{}|{}] val_loss: {} | val_cosine: {}'.format(epoch, self.epochs, val_loss, val_cosine.mean()))

                if counter >= self.patience:
                    print('EarlyStopping counter: {}'.format(counter))
                    break

        print('best epoch [{}|{}] val_ loss: {} | val_cosine: {}'.format(self.bestepoch, self.epochs,
                                                                         self.val_loss_best, self.val_cosine_best))
        self.traininfor = {'val_losses': val_losses, 'val_cosines': val_cosines}


    def predict(self, evidence, msms):
        dfdb = deepcopy(evidence)
        msms = deepcopy(msms).rename(columns={'id': 'Best MS/MS'})
        dfdb['CMS'] = dfdb['Charge'].map(str) + dfdb['Modified sequence']

        # dfdb['CMS'] = dfdb['CMS'].apply(lambda x: x.replace('_(ac)M(ox)', 'B').replace(
        #     '_(ac)', 'J').replace('M(ox)', 'O').replace('_', ''))

        dfdb['CMS'] = dfdb['CMS'].apply(
            lambda x: x.replace('(Acetyl (Protein N-term))M(Oxidation (M))', 'B').replace(
                '(Acetyl (Protein N-term))', 'J').replace('(Oxidation (M))', 'O').replace('_', ''))

        dfdb = pd.merge(dfdb, msms[['Best MS/MS', 'Matches', 'Intensities']], on='Best MS/MS', how='left')
        dfdb1 = deepcopy(dfdb[(~dfdb['Matches'].isna()) &
                              (~dfdb['Intensities'].isna()) &
                              (dfdb['Length'] <= 47) &
                              (dfdb['Charge'] <= 6)])[['id', 'CMS', 'Matches', 'Intensities']]
        print('after filter none Intensities data shape:', dfdb1.shape)  
        print('+++++++++++++++++++Loading Testset+++++++++++++++++++++')
        x_test, y_test = IonCoding().fit_transfrom(dfdb1[['CMS', 'Matches', 'Intensities']])
        self.db_test = {'Data': dfdb1, 'x_test': x_test, 'y_test': y_test}
        x_test = torch.tensor(x_test, dtype=torch.float)
        test_loader = DataLoader(x_test,
                                 batch_size=self.batch_size,
                                 num_workers=0,
                                 shuffle=False)
        print('+++++++++++++++++++DeepSpec Testing+++++++++++++++++++++')
        y_testpreds = []
        model = torch.load('DeepSpec_nanowell2.pkl').to(self.device)

        model.eval()
        with torch.no_grad():
            for test_x in test_loader:
                test_x = test_x.to(self.device)
                y_testpreds.append(model(test_x))
            y_testpred = torch.cat(y_testpreds, dim=0)

        y_test = np.array(y_test).reshape(y_test.shape[0], -1)
        y_testpred = np.array(y_testpred.cpu())
        self.db_test['y_testpred'] = y_testpred
        y_testpred = y_testpred.reshape(y_test.shape[0], -1)
        CS = self.cosine_similarity(y_test, y_testpred)
        self.db_test['Cosine'] = CS
        output = pd.DataFrame({'id': dfdb1['id'].values, 'Cosine': CS})
        dfdb2 = pd.merge(dfdb, output, on='id', how='left')
        dfdb2['PEPCosine'] = dfdb2['Cosine'] / (1 + dfdb2['PEP'])
        dfdb2['ScoreCosine'] = dfdb2['Score'] / (1 + dfdb2['Cosine'])
        return dfdb2

    def cosine_similarity(self, y, y_pred):
        a, b = np.array(y), np.array(y_pred)
        res = np.array([[sum(a[i] * b[i]), np.sqrt(sum(a[i] * a[i]) * sum(b[i] * b[i]))]
                        for i in range(a.shape[0])])
        return np.divide(res[:, 0], res[:, 1])  # Cosine or DP
        # return 1 - 2 * np.arccos(np.divide(res[:, 0], res[:, 1])) / np.pi  # SA

    def calculate_mse(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        squared_errors = (y_true - y_pred) ** 2

        mse = np.mean(squared_errors)

        return mse

    def calculate_mae(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        absolute_errors = np.abs(y_true - y_pred)

        mae = np.mean(absolute_errors)

        return mae

    def calculate_pearson(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        pearson_coefficients = []
        for i in range(y_true.shape[0]):
            corr, _ = pearsonr(y_true[i], y_pred[i])
            pearson_coefficients.append(corr)

        return np.mean(np.array(pearson_coefficients))

    def frobenius_norm_of_difference(self, A, B):

        A = np.array(A)
        B = np.array(B)

        C = A - B

        frobenius_norm = np.linalg.norm(C, 'fro')

        return frobenius_norm

    def selectBestmsms(self, df, lg=47, cg=6, s=100):
        return df[(df['Reverse'] != '+') & (~df['Matches'].isna()) &
                  (~df['Intensities'].isna()) & (df['Length'] <= lg) &
                  (df['Charge'] <= cg) & (df['Type'].isin(['MSMS', 'MULTI-MSMS']))
                  & (df['Score'] > s)].sort_values(
            'Score', ascending=False).drop_duplicates(
            subset='CMS', keep='first')[['CMS', 'Matches', 'Intensities']]

    def ValPlot(self):
        val_losses = self.traininfor['val_losses']
        val_cosines = self.traininfor['val_cosines']
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        lns1 = ax.plot(range(1, len(val_losses) + 1), val_losses,
                       color='orange', label='Val Loss={}'.format(round(self.val_loss_best, 5)))
        lns2 = ax.axvline(x=self.bestepoch, ls="--", c="b", label='kk')
        plt.xticks(size=15)
        plt.yticks(size=15)

        ax2 = ax.twinx()
        lns3 = ax2.plot(range(1, len(val_losses) + 1), [i.mean() for i in val_cosines],
                        color='red', label='Val Cosine={}'.format(round(self.val_cosine_best, 4)))

        lns = lns1 + lns3
        labs = [l.get_label() for l in lns]

        plt.yticks(size=15)
        ax.set_xlabel("Epoch", fontsize=18)
        ax.set_ylabel("Val Loss", fontsize=18)
        ax2.set_ylabel("Val Cosine", fontsize=18)
        ax.legend(lns, labs, loc=10, fontsize=15)
        plt.tight_layout()

class LGB_bayesianCV:
    def __init__(self, params_init=dict({}), n_splits=3, seed=0):
        self.n_splits = n_splits
        self.seed = seed
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'n_jobs': -1,
            'random_state': self.seed,
            'is_unbalance': True,
            'silent': True
        }
        self.params.update(params_init)

    def fit(self, x, y):
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.__lgb_bayesian()
        self.model = lgb.LGBMClassifier(**self.params)
        # self.cv_predprob = cross_val_predict(self.model, self.__x, self.__y,
        #                                      cv=self.skf,  method="predict_proba")[:, 1]
        self.model.fit(self.__x, self.__y)
        # self.feature_importance = dict(zip(self.model.feature_name_, self.model.feature_importances_))

    def predict(self, X):
        return self.model.predict(np.array(X))

    def predict_proba(self, X):
        return self.model.predict_proba(np.array(X))

    def __lgb_cv(self, n_estimators, learning_rate,
                 max_depth, num_leaves,
                 subsample, colsample_bytree,
                 min_split_gain, min_child_samples,
                 reg_alpha, reg_lambda):
        self.params.update({
            'n_estimators': int(n_estimators),  #  (100, 1000)
            'learning_rate': float(learning_rate),  #  (0.001, 0.3)
            'max_depth': int(max_depth),  #  (3, 15)
            'num_leaves': int(num_leaves),  # (2, 2^md) (5, 1000)
            'subsample': float(subsample),  #  (0.3, 0.9)
            'colsample_bytree': float(colsample_bytree),  #  (0.3, 0.9)
            'min_split_gain': float(min_split_gain),  #  (0, 0.5)
            'min_child_samples': int(min_child_samples),  # (5, 1000)
            'reg_alpha': float(reg_alpha),  #  (0, 10)
            'reg_lambda': float(reg_lambda),  #  (0, 10)
        })

        model = lgb.LGBMClassifier(**self.params)
        cv_score = cross_val_score(model, self.__x, self.__y, scoring="roc_auc", cv=self.skf).mean()
        return cv_score

    def __lgb_bayesian(self):
        lgb_bo = BayesianOptimization(self.__lgb_cv,
                                      {
                                          'n_estimators': (100, 1000),  #  (100, 1000)
                                          'learning_rate': (0.001, 0.3),  #  (0.001, 0.3)
                                          'max_depth': (3, 15),  #  (3, 15)
                                          'num_leaves': (5, 1000),  #  (2, 2^md) (5, 1000)
                                          'subsample': (0.3, 0.9),  #  (0.3, 0.9)
                                          'colsample_bytree': (0.3, 0.9),  # 
                                          'min_split_gain': (0, 0.5),  #  (0, 0.5)
                                          'min_child_samples': (5, 200),  # (5, 1000)
                                          'reg_alpha': (0, 10),  #  (0, 10)
                                          'reg_lambda': (0, 10),  #  (0, 10)
                                      },
                                      random_state=self.seed,
                                      verbose=0)
        lgb_bo.maximize()
        self.best_auc = lgb_bo.max['target']
        lgbbo_params = lgb_bo.max['params']
        lgbbo_params['n_estimators'] = int(lgbbo_params['n_estimators'])
        lgbbo_params['learning_rate'] = float(lgbbo_params['learning_rate'])
        lgbbo_params['max_depth'] = int(lgbbo_params['max_depth'])
        lgbbo_params['num_leaves'] = int(lgbbo_params['num_leaves'])
        lgbbo_params['subsample'] = float(lgbbo_params['subsample'])
        lgbbo_params['colsample_bytree'] = float(lgbbo_params['colsample_bytree'])
        lgbbo_params['min_split_gain'] = float(lgbbo_params['min_split_gain'])
        lgbbo_params['min_child_samples'] = int(lgbbo_params['min_child_samples'])
        lgbbo_params['reg_alpha'] = float(lgbbo_params['reg_alpha'])
        lgbbo_params['reg_lambda'] = float(lgbbo_params['reg_lambda'])
        self.params.update(lgbbo_params)


class LgbBayes:
    def __init__(self, out_cv=3, inner_cv=3, seed=0):
        self.out_cv = out_cv
        self.inner_cv = inner_cv
        self.seed = seed

    def fit_tranform(self, data, feature_columns, target_column, file_column, protein_column=None):
        data_set = deepcopy(data)
        x = deepcopy(data_set[feature_columns]).values
        y = deepcopy(data_set[target_column]).values#label 
        skf = StratifiedKFold(n_splits=self.out_cv, shuffle=True, random_state=self.seed)
        cv_index = np.zeros(len(y), dtype=int)#
        y_prob = np.zeros(len(y))#
        y_pep = np.zeros(len(y))#
        feature_importance_df = pd.DataFrame()

        for index, (train_index, test_index) in enumerate(skf.split(x, y)):
            print('++++++++++++++++CV {}+++++++++++++++'.format(index + 1))
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lgbbo = LGB_bayesianCV(n_splits=self.inner_cv)
            lgbbo.fit(x_train, y_train)

            y_testprob = lgbbo.predict_proba(x_test)[:, 1]
            y_prob[test_index] = y_testprob
            cv_index[test_index] = index + 1
            print('train auc:', lgbbo.best_auc)  # best val auc  lgbbo.model.best_score
            print('test auc:', roc_auc_score(y_test, y_testprob)) #
            y_pep[test_index] = Prob2PEP(y_test, y_testprob)

            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = feature_columns
            fold_importance_df["Importance"] = lgbbo.model.feature_importances_
            fold_importance_df["cv_index"] = index + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        y_qvalue = Score2Qval(y, -y_pep) 
        self.feature_imp = feature_importance_df[['Feature', 'Importance']].groupby(
            'Feature').mean().reset_index().sort_values(by='Importance')
        data_set['cv_index'] = cv_index
        data_set['Lgb_score'] = y_prob
        data_set['Lgb_pep'] = y_pep
        data_set['psm_qvalue'] = y_qvalue
        if protein_column is not None:
            data_set = GroupProteinPEP2Qval(data_set, file_column=file_column,
                                            protein_column=protein_column,
                                            target_column=target_column,
                                            pep_column='Lgb_pep')
        self.data_set = data_set
        return data_set

    def Feature_imp_plot(self):
        plt.figure(figsize=(6, 6))
        plt.barh(self.feature_imp.Feature, self.feature_imp.Importance, height=0.7, orientation="horizontal")
        plt.ylim(0, self.feature_imp.Feature.shape[0] - 0.35)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('Importance', fontsize=15)
        plt.ylabel('Features', fontsize=15)
        plt.tight_layout()

    def CVROC(self):
        index_column = 'cv_index'
        target_column = 'label'
        pred_column = 'Lgb_score'
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 10000)
        plt.figure(figsize=(5, 4))
        for i in sorted(self.data_set[index_column].unique()):
            y_true = self.data_set.loc[self.data_set[index_column] == i, target_column]
            y_prob = self.data_set.loc[self.data_set[index_column] == i, pred_column]

            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            mean_tpr += sp.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='Fold{} AUC = {}'.format(i, round(roc_auc, 4)))

        mean_tpr = mean_tpr / len(self.data_set[index_column].unique())
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label='Mean AUC = {}'.format(round(mean_auc, 4)))
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('True Positive Rate', fontsize=15)
        plt.legend(fontsize=12)
        plt.tight_layout()

    def PSM_accept(self):
        data = self.data_set[self.data_set['psm_qvalue'] <= 0.05]
        data = data.sort_values(by='psm_qvalue')
        data['Number of PSMs'] = range(1, data.shape[0] + 1)

        plt.figure(figsize=(5, 4))
        plt.plot(data['psm_qvalue'], data['Number of PSMs'], label='DeepSCP')
        plt.axvline(x=0.01, ls="--", c="gray")
        plt.xlim(-0.001, 0.05)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('PSM q-value', fontsize=15)
        plt.ylabel('Number of PSMs', fontsize=15)
        plt.legend(fontsize=13)
        plt.tight_layout()
 
class IonCoding:
    def __init__(self, bs=1000, n_jobs=-1):
        ino = [
            '{0}{2};{1}{2};{0}{2}(2+);{1}{2}(2+);{0}{2}-NH3;{1}{2}-NH3;{0}{2}(2+)-NH3;{1}{2}(2+)-NH3;{0}{2}-H20;{1}{2}-H20;{0}{2}(2+)-H20;{1}{2}(2+)-H20'.
            format('b', 'y', i) for i in range(1, 47)]
        ino = np.array([i.split(';') for i in ino]).flatten()
        self.MI0 = pd.DataFrame({'MT': ino})
        self.bs = bs
        self.n_jobs = n_jobs

    def fit_transfrom(self, data):
        print('++++++++++++++++OneHotEncoder CMS(Chage + Modified sequence)++++++++++++++')
        t0 = time()

        x = self.onehotfeature(data['CMS']).reshape(data.shape[0], -1, 30)

        print('using time', time() - t0)
        print('x shape: ', x.shape)
        print('++++++++++++++++++++++++++Construct Ion Intensities Array++++++++++++++++++++')
        t0 = time()
        y = self.ParallelinoIY(data[['Matches', 'Intensities']])
        print('using time', time() - t0)
        print('y shape: ', y.shape)
        return x, y

    def onehotfeature(self, df0, s=48):
        df = df0.apply(lambda x: x + (s - len(x)) * 'Z')

        # B: '_(ac)M(ox)'; 'J': '_(ac)'; 'O': 'M(ox)'; 'Z': None
        aminos = '123456ABCDEFGHIJKLMNOPQRSTVWYZ'
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(np.repeat(np.array(list(aminos)), s).reshape(-1, s))

        seqs = np.array(list(df.apply(list)))

        return enc.transform(seqs).toarray().reshape(df.shape[0], -1, 30)

    def ParallelinoIY(self, data):
        datasep = [data.iloc[i * self.bs: (i + 1) * self.bs] for i in range(data.shape[0] // self.bs + 1)]

        paraY = Parallel(n_jobs=self.n_jobs)(delayed(self.dataapp)(i) for i in datasep)

        return np.vstack(paraY).reshape(data.shape[0], -1, 12)

    def inoIY(self, x):
        MI = pd.DataFrame({'MT0': x[0].split(';'), 'IY': [float(i) for i in x[1].split(';')]})
        dk = pd.merge(self.MI0, MI, left_on='MT', right_on='MT0', how='left').drop('MT0', axis=1)
        dk.loc[dk.IY.isna(), 'IY'] = 0
        dk['IY'] = dk['IY'] / dk['IY'].max()
        return dk['IY'].values

    def dataapp(self, data):
        return np.array(list(data.apply(self.inoIY, axis=1)))


class CNN_BiLSTM(nn.Module):
    def __init__(self, batchsize, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(CNN_BiLSTM, self).__init__()
        self.device = device
        self.batchsize = batchsize
        self.cov = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                      out_channels=64,
                      kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5))

        #self.lstm = CustomBiLSTM(batchsize=batchsize, input_dim=64, hidden_dim=hidden_dim, num_layers=layer_dim,
                                 # device=torch.device('cuda:0'))
        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=hidden_dim,
                            num_layers=layer_dim,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.5)
        self.mambaBlock11 = Mamba(d_model=64, d_conv=3, d_state =16).to(device)

        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, output_dim),
                                nn.Sigmoid())

    def forward(self, x):


        x = self.cov(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.mambaBlock11(x)
        l_out, (l_hn, l_cn) = self.lstm(x, None)

        x = self.fc(l_out)  # 46*12
        # print('asdadfs4', x.shape)

        return x


class CustomBiLSTM(nn.Module):
    def __init__(self, batchsize, input_dim, hidden_dim, num_layers, device, dropout=0.5):
        super(CustomBiLSTM, self).__init__()
        self.batchsize = batchsize
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                                  dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.mambaBlock1 = Mamba(d_model=self.batchsize, d_conv=2)

        self.reset_parameters()

    def lstm_forward(self, input, initial_states, w_ih, w_hh, b_ih, b_hh):
  

        h_0, c_0 = initial_states  # 
        batch_size, T, input_size = input.shape  # T
        hidden_size = w_ih.shape[0] // 4
        prev_h = h_0
        prev_c = c_0
        batch_w_ih = w_ih.unsqueeze(0).tile(batch_size, 1, 1)  # [batch_size, 4*hidden_size, input_size]
        batch_w_hh = w_hh.unsqueeze(0).tile(batch_size, 1, 1)  # [batch_size, 4*hidden_size, hidden_size]

        output_size = hidden_size
        output = torch.zeros(batch_size, T, output_size)  # 

        for t in range(T):
            x = input[:, t, :]  # [batch_size*input_size]
            w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))  # [batch_size, 4*hidden_size, 1],[2,80,1]
            w_times_x = w_times_x.squeeze(-1)  # [batch_size, 4*hidden_size],[2,80]

            w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))  # [batch_size, 4*hidden_size, 1]
            w_times_h_prev = w_times_h_prev.squeeze(-1)  # [batch_size, 4*hidden_size]


            # it
            # i_t = torch.sigmoid(w_times_x[:, :hidden_size] + w_times_h_prev[:, :hidden_size] + b_ih[:hidden_size] + b_hh[:hidden_size])
            Wi_t = w_times_x[:, :hidden_size] + w_times_h_prev[:, :hidden_size]
            Wi_t_expanded = Wi_t.unsqueeze(1)
            Wi_t = self.mambaBlock1(Wi_t_expanded).squeeze(1)
            i_t = torch.sigmoid(Wi_t + b_ih[:hidden_size] + b_hh[:hidden_size])

            # ft
            # f_t = torch.sigmoid(
            #     w_times_x[:, hidden_size:2 * hidden_size] + w_times_h_prev[:, hidden_size:2 * hidden_size]
            #     + b_ih[hidden_size:2 * hidden_size] + b_hh[hidden_size:2 * hidden_size])
            Wf_t = w_times_x[:, hidden_size:2 * hidden_size] + w_times_h_prev[:, hidden_size:2 * hidden_size]
            f_t = torch.sigmoid(Wf_t + b_ih[hidden_size:2 * hidden_size] + b_hh[hidden_size:2 * hidden_size])

            # C-hat
            g_t = torch.tanh(
                w_times_x[:, 2 * hidden_size:3 * hidden_size] + w_times_h_prev[:, 2 * hidden_size:3 * hidden_size]
                + b_ih[2 * hidden_size:3 * hidden_size] + b_hh[2 * hidden_size:3 * hidden_size])

            # Ot
            # o_t = torch.sigmoid(
            #     w_times_x[:, 3 * hidden_size:4 * hidden_size] + w_times_h_prev[:, 3 * hidden_size:4 * hidden_size]
            #     + b_ih[3 * hidden_size:4 * hidden_size] + b_hh[3 * hidden_size:4 * hidden_size])
            WO_t = w_times_x[:, 3 * hidden_size:4 * hidden_size] + w_times_h_prev[:, 3 * hidden_size:4 * hidden_size]
            o_t = torch.sigmoid(WO_t + b_ih[3 * hidden_size:4 * hidden_size] + b_hh[3 * hidden_size:4 * hidden_size])

            # Ct
            prev_c = f_t * prev_c + i_t * g_t
            # ht
            prev_h = o_t * torch.tanh(prev_c)

            output[:, t, :] = prev_h
            output = output.to(self.device)

        return output, (prev_h, prev_c)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):

        # x  (batch_size, seq_length, input_dim)
        batch_size, T, input_size = x.shape[0], x.shape[1], x.shape[2]
        h_0_fw = torch.zeros(batch_size, self.hidden_dim).to(self.device)  # 
        c_0_fw = torch.zeros(batch_size, self.hidden_dim).to(self.device)  # 
        h_0_bw = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_0_bw = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        # print('111',h_0_fw.device,c_0_fw.device,h_0_bw.device,c_0_bw.device)

        # output_custom = x
        out = x

        for layer in range(self.num_layers):
            output_custom = out

            w_ih_fw = getattr(self.lstm_layer, f'weight_ih_l{layer}')  # .to(self.device)
            w_hh_fw = getattr(self.lstm_layer, f'weight_hh_l{layer}')  # .to(self.device)
            b_ih_fw = getattr(self.lstm_layer, f'bias_ih_l{layer}')  # .to(self.device)
            b_hh_fw = getattr(self.lstm_layer, f'bias_hh_l{layer}')  # .to(self.device)

            w_ih_bw = getattr(self.lstm_layer, f'weight_ih_l{layer}_reverse')  # .to(self.device)
            w_hh_bw = getattr(self.lstm_layer, f'weight_hh_l{layer}_reverse')  # .to(self.device)
            b_ih_bw = getattr(self.lstm_layer, f'bias_ih_l{layer}_reverse')  # .to(self.device)
            b_hh_bw = getattr(self.lstm_layer, f'bias_hh_l{layer}_reverse')  # .to(self.device)
            output_custom_b = out
            x_reversed = torch.flip(output_custom_b, [1])  # .to(self.device)  # 

            # output_custom=output_custom.to(self.device)
            output_custom_fw, (h_0_fw, c_0_fw) = self.lstm_forward(
                input=output_custom,
                initial_states=(h_0_fw, c_0_fw),
                w_ih=w_ih_fw,
                w_hh=w_hh_fw,
                b_ih=b_ih_fw,
                b_hh=b_hh_fw
            )

            output_custom_bw, (h_0_bw, c_0_bw) = self.lstm_forward(
                input=x_reversed,
                initial_states=(h_0_bw, c_0_bw),
                w_ih=w_ih_bw,
                w_hh=w_hh_bw,
                b_ih=b_ih_bw,
                b_hh=b_hh_bw
            )

            output_custom_b = torch.flip(output_custom_bw, [1])  # 
            out = torch.cat((output_custom_fw, output_custom_b),
                            dim=-1)  # out (batch_size, seq_length, 2 * hidden_dim)
            if layer < self.num_layers - 1:
                out = self.dropout(out)

        return out


# endregion
 
 


def PSM2ProPep(data, file_column,
               protein_column,
               peptide_column,
               intensity_columns):
    df = data[[file_column] + [protein_column] + [peptide_column] + intensity_columns]

    proteins = df.groupby([file_column, protein_column])[intensity_columns].sum(1).reset_index()
    df_pro = []
    for i, j in proteins.groupby([file_column]):
        k = pd.DataFrame(data=j[intensity_columns].values, index=j[protein_column].tolist(),
                         columns=['{}_{}'.format(i, x) for x in intensity_columns])
        df_pro.append(k)
    df_pro = pd.concat(df_pro, axis=1)
    df_pro[df_pro.isna()] = 0
    df_pro.index.name = protein_column

    peptides = df.groupby([file_column, protein_column, peptide_column])[intensity_columns].sum(1).reset_index()
    df_pep = []
    for i, j in peptides.groupby([file_column]):
        k = j.drop(file_column, axis=1).set_index([protein_column, peptide_column])
        k.columns = ['{}_{}'.format(i, x) for x in intensity_columns]
        df_pep.append(k)
    df_pep = pd.concat(df_pep, axis=1)
    df_pep[df_pep.isna()] = 0
    df_pep = df_pep.reset_index()
    return df_pro, df_pep


def proteinfilter(data, protein_count=15, sample_ratio=0.5):
    nrow = (data != 0).sum(1)
    data = data.loc[nrow[nrow >= protein_count].index]
    ncol = (data != 0).sum(0)
    data = data[ncol[ncol >= ncol.mean() * sample_ratio].index]
    return data



#endregion

def main(evidenve_file, msms_file, lbmsms_file):
    folder_path = '/home/DeepSCP-main/inputData/nanowell/SampleSet'
    print(' ###################SampleRT###################')
    # region SampleRT module(Sample set)
    full_path = os.path.join(folder_path, evidenve_file)
    evidence = pd.read_csv(full_path, sep='\t', low_memory=False)  # sample
    sampleRT = MQ_SampleRT()
    dfRT = sampleRT.fit_tranform(evidence)

    # region 
    SampleRT_scores = sampleRT.cmp_scores
    SampleRT_scores.to_csv('./outputData/nanowell/SampleRT_scores.csv', index=False)
    plt.figure(figsize=(5, 4))
    # SampleRT_scores = pd.read_csv('../input/SampleRT_scores.csv')
    SampleRT_scores = SampleRT_scores[SampleRT_scores['score'] > 0]

 
    # endregion
    del evidence
    # endregion


    dfRT.to_csv('./outputData/nanowell/dfRT_.csv', index=None)

    #folder_path = '/home/DeepSCP-main/inputData/nanowell/SampleSet'
    full_path2 = os.path.join(folder_path, msms_file)
    folder_path2 = '/home/DeepSCP-main/inputData/nanowell/LibrarySet'
    full_path3 = os.path.join(folder_path2, lbmsms_file)
    msms = pd.read_csv(full_path2, sep='\t', low_memory=False)  # sample
    lbmsms = pd.read_csv(full_path3, sep='\t', low_memory=False)  # lib
    deepspec = DeepSpec()
    deepspec.fit(lbmsms)  

    # region 
    deepspec.ValPlot()
    plt.savefig('./outputData/nanowell/figure/DeepSpec.pdf')
    deepspec.traininfor.keys()
    # endregion

    dfSP = deepspec.predict(dfRT, msms)  
    deepspec.db_test.keys()
    del msms, lbmsms
    dfSP.to_csv('./outputData/nanowell/dfSP_.csv', index=None)

    print('###################LgbBayses###################')
    # region LgBayses
    dfdb = deepcopy(dfSP)
    del dfSP

    feature_columns = ['Length', 'Acetyl (Protein N-term)', 'Oxidation (M)', 'Missed cleavages',
                       'Charge', 'm/z', 'Mass', 'Mass error [ppm]', 'Retention length', 'PEP',
                       'MS/MS scan number', 'Score', 'Delta score', 'PIF', 'Intensity',
                       'Retention time', 'RT(*|rev)', 'RT(*|tag)', 'DeltaRT', 'PEPRT', 'ScoreRT',
                       'Cosine', 'PEPCosine', 'ScoreCosine']
    target_column = 'label'
    file_column = 'Experiment'
    protein_column = 'Leading razor protein'
    lgs = LgbBayes()

    data_set = lgs.fit_tranform(data=dfdb,
                                feature_columns=feature_columns,
                                target_column=target_column,
                                file_column=file_column,
                                protein_column=protein_column)

    # region 
    feature_imp = lgs.feature_imp
    feature_imp.to_csv('./outputData/nanowell/feature_imp.csv', index=None)
    lgs.Feature_imp_plot()
    plt.savefig('./outputData/nanowell/figure/Feature_imp.pdf')
    lgs.CVROC()
    plt.savefig('./outputData/nanowell/figure/DeepSCP_ROC.pdf')
    lgs.PSM_accept()
    plt.savefig('./outputData/nanowell/figure/PSM_accept.pdf')
    # endregion

    data = data_set[(data_set.psm_qvalue < 0.01) & (data_set.protein_qvalue < 0.01) &
                    (data_set.label == 1)]

    peptide_column = 'Sequence'
    intensity_columns = [i for i in data.columns if 'Reporter intensity corrected' in i]

    df_pro, df_pep = PSM2ProPep(data, file_column=file_column,
                                protein_column=protein_column,
                                peptide_column=peptide_column,
                                intensity_columns=intensity_columns)

    data_set.to_csv('./outputData/nanowell/DeepSCP_evidence.txt', sep='\t', index=False)
    data.to_csv('./outputData/nanowell/DeepSCP_evidence_filter.txt', sep='\t', index=False)

    df_pro.to_csv('./outputData/nanowell/DeepSCP_pro.csv')
    df_pep.to_csv('./outputData/nanowell/DeepSCP_pep.csv', index=False)
    # endregion
    data.to_csv('./outputData/nanowell/data_.csv', index=None)
    data_set.to_csv('./outputData/nanowell/data_set_.csv', index=None)

    print('###################Protein filter###################')

    # region unknow module
    an_cols = pd.DataFrame({'Sample_id': df_pro.columns,
                            'Set': [i.rsplit('_', 1)[0] for i in df_pro.columns],
                            'Channel': [i.rsplit('_', 1)[-1] for i in df_pro.columns]})
    an_cols['Cleanset'] = an_cols['Set'].str.strip("()'',")

    an_cols['Type'] = 'Empty'
    an_cols.loc[an_cols.Channel == 'Reporter intensity corrected 10', 'Type'] = 'Boost'
    an_cols.loc[an_cols.Channel == 'Reporter intensity corrected 7', 'Type'] = 'Reference'

    # region nano

    # 126
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['1_A', '1_B', '1_C', '2_A', '2_B', '2_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 1'), 'Type'] = 'C10'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['3_A', '3_B', '3_C', '4_A', '4_B', '4_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 1'), 'Type'] = 'SVEC'

    # 127N
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['1_A', '1_B', '1_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 2'), 'Type'] = 'C10'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['2_A', '2_B', '2_C', '3_A', '3_B', '3_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 2'), 'Type'] = 'SVEC'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['4_A', '4_B', '4_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 2'), 'Type'] = 'RAW'

    # 127C
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['1_A', '1_B', '1_C', '2_A', '2_B', '2_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 3'), 'Type'] = 'SVEC'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['3_A', '3_B', '3_C', '4_A', '4_B', '4_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 3'), 'Type'] = 'RAW'

    # 128N
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['1_A', '1_B', '1_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 4'), 'Type'] = 'SVEC'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['2_A', '2_B', '2_C', '3_A', '3_B', '3_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 4'), 'Type'] = 'RAW'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['4_A', '4_B', '4_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 4'), 'Type'] = 'C10'

    # 128C
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['1_A', '1_B', '1_C', '2_A', '2_B', '2_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 5'), 'Type'] = 'RAW'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['3_A', '3_B', '3_C', '4_A', '4_B', '4_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 5'), 'Type'] = 'C10'

    # 129N
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['1_A', '1_B', '1_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 6'), 'Type'] = 'RAW'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['2_A', '2_B', '2_C', '3_A', '3_B', '3_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 6'), 'Type'] = 'C10'
    an_cols.loc[(an_cols['Cleanset'].isin(
        ['4_A', '4_B', '4_C'])) &
                (an_cols['Channel'] == 'Reporter intensity corrected 6'), 'Type'] = 'SVEC'
    # endregion

    an_cols.to_csv('./outputData/nanowell/an_cols.csv', index=False)
    an_cols1 = an_cols[(an_cols.Type.isin(['C10', 'SVEC', 'RAW']))]
    an_cols1.Type.value_counts()
    df_pro1 = df_pro[list(set(df_pro.columns) & set(an_cols1.Sample_id))]

    # df_pro1 = df_pro[set(df_pro.columns) & set(an_cols1.Sample_id)]
    df_pep1 = df_pep[[protein_column] + [peptide_column] + df_pro1.columns.tolist()]

    df_pro_ft = proteinfilter(df_pro1)
    df_pep_ft = df_pep1[df_pep1[protein_column].isin(df_pro_ft.index)]
    (df_pro_ft != 0).sum(0).mean()
    (df_pep_ft.iloc[:, 2:] != 0).sum(0).mean()
    df_pro_ft.to_csv('./outputData/nanowell/DeepSCP_pro_ft.csv')
    df_pep_ft.to_csv('./outputData/nanowell/DeepSCP_pep_ft.csv')
    # endregion

    print('ok----------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepSCP: utilizing deep learning to boost single-cell proteome coverage")
    parser.add_argument("-e",
                        "--evidence",
                        dest='e',
                        type=str,
                        help="SCP SampleSet, evidence.txt, which recorde information about the identified peptides \
                        by MaxQuant with setting  FDR to 1 at both PSM and protein levels")
    parser.add_argument("-m",
                        "--msms",
                        dest='m',
                        type=str,
                        help="SCP SampleSet, msms.txt, which recorde fragment ion information about the identified peptides \
                        by MaxQuant with setting  FDR to 1 at both PSM and protein levels")
    parser.add_argument("-lbm",
                        "--lbmsms",
                        dest='lbm',
                        type=str,
                        help="LibrarySet, msms.txt, which recorde fragment ion information about the identified peptides \
                        by MaxQuant with setting  FDR to 0.01 at both PSM and protein levels")
    args = parser.parse_args()
    evidenve_file = args.e
    msms_file = args.m
    lbmsms_file = args.lbm
    t0 = time()
    main(evidenve_file, msms_file, lbmsms_file)
    print('DeepSCP using time: {} m {}s'.format(int((time() - t0) // 60), (time() - t0) % 60))
