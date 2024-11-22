import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from numba import njit, prange
import time

from scipy import stats
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as ssd


@njit
def mean_dist_pairwise(matrix, shape):
    dist = np.zeros((shape, shape))

    for i in prange(shape):
        for j in prange(shape):
            dist[j, i] = np.nanmean(matrix[:, i] - matrix[:, j])

    return dist



# Load evidence.txt files from folder
filenames = glob.glob("../data/evidence*.txt")
evidences = [pd.read_csv(filename, sep='\t', engine='python', header=0) for filename in filenames]

# Combine all evidences in one dataframe
evidence_all = pd.concat(evidences, sort=False, ignore_index=True)
# Clean up
del evidences

evidence_all = evidence_all.loc[(evidence_all['Reverse'] != '+') & \
                                 (evidence_all['Intensity'] > 0) & \
                                 (evidence_all['Charge'] != 1)]


# Keep only one evidence per raw file
# Select feature with maximum intensity

selection = ['Modified sequence', 'Sequence', 'Charge', 'Mass', 'm/z', 'CCS', 'Experiment',
             'id', 'Intensity', 'Score', 'Length', 'Raw file']
evidence_agg = evidence_all.loc[evidence_all.groupby(
    ['Modified sequence', 'Charge', 'Raw file'])['Intensity'].idxmax()][selection]

evidence_pivot_long = evidence_agg.pivot_table(index = ['Modified sequence', 'Charge'],
                                               columns = 'Raw file',
                                               values = 'CCS')
del evidence_agg

evidence_pivot_long = evidence_pivot_long.astype(np.float32)
print('ev',evidence_all.shape)
print('ev',evidence_pivot_long.shape)

# Filter peptides with only one occurence to speed up and save memory (747 runs - 1)

evidence_pivot = evidence_pivot_long.loc[evidence_pivot_long.isnull().sum(axis=1) < (len(set(evidence_all['Raw file'])) - 1)]
evidence_pivot = evidence_pivot.astype(np.float32)


# calculate pair-wise distances
evidence_distance = pd.DataFrame(mean_dist_pairwise(np.array(evidence_pivot), evidence_pivot.shape[1]))
# Fill NA distances with arbitrary large number
evidence_distance_cluster = evidence_distance.fillna(3*evidence_distance.max().max())


# save result matrix
evidence_distance.to_csv('output2/evidence_distance_float32.csv', index=False)
dist_matrix = ssd.squareform(evidence_distance_cluster, checks=False)


# Perform hierarchical clustering to sort runs by distance
Z = shc.linkage(abs(dist_matrix), 'ward')
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
shc.dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=250,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.savefig("output2/layerClustering_1.png")
plt.close()

nruns = len(evidence_pivot_long.columns)
col = nruns
nancount = 0

for cluster in Z:
    # loop through clusters and merge them pairwise

    # align first
    delta = np.nanmean(evidence_pivot_long.iloc[:, int(cluster[1])] - evidence_pivot_long.iloc[:, int(cluster[0])])

    if (~np.isnan(delta)):
        # Merge neighboring runs
        # Calculate mean

        evidence_pivot_long[col] = np.nanmean([evidence_pivot_long.iloc[:, int(cluster[1])] - 0.5 * delta,
                                               evidence_pivot_long.iloc[:, int(cluster[0])] + 0.5 * delta], axis=0)
    else:
        # NaN difference in a cluster to be merged
        # Merge runs but keep external calibration

        evidence_pivot_long[col] = np.nanmean([evidence_pivot_long.iloc[:, int(cluster[1])],
                                               evidence_pivot_long.iloc[:, int(cluster[0])]], axis=0)

    col += 1

# Store experimental values in separate dataframe
evidence_pivot_tmp = evidence_pivot_long.copy()
evidence_pivot_long = evidence_pivot_long.iloc[:, 0:nruns]



# Calculate deviation from mean to use as correction factors

evidence_pivot_deviation = evidence_pivot_long.subtract(evidence_pivot_tmp.iloc[:, -1], axis = 0)
evidence_pivot_deviation.mean(axis = 0, skipna = True).plot(figsize = (30,8))

plt.xlabel('Raw file')
plt.ylabel('Mean deviation from the median CCS')
plt.xticks(size=8, rotation=10);
plt.savefig("output2/MeandeviationCCS_2.png")
plt.close()


evidence_pivot_aligned = evidence_pivot_long.subtract(evidence_pivot_deviation.mean(axis = 0, skipna = True), axis = 1)

# Export aligned dataset
evidence_pivot_aligned.to_csv('output2/evidence_pivot_aligned_3.csv', index=False)


# Exclude Proteome Tools data
# proteometools = set(evidence_all.loc[evidence_all['Experiment'].isin(
#     ['Proteotypic', 'SRMATLAS', 'MissingGeneSet'])]['Raw file'])
keywords = ['Proteotypic', 'SRMATLAS', 'MissingGeneSet']
pattern = '|'.join(keywords)
proteometools = set(evidence_all.loc[evidence_all['Experiment'].str.contains(pattern, na=False)]['Raw file'])

evidence_pivot_aligned_endo = evidence_pivot_aligned.drop(proteometools, axis = 1)

# Calculate mean CCS value from aligned data
evidence_short_aligned = evidence_pivot_aligned_endo.drop(columns = evidence_pivot_aligned_endo.columns.tolist())
evidence_short_aligned['CCS'] = evidence_pivot_aligned_endo.mean(axis = 1, skipna = True)
evidence_short_aligned.reset_index(inplace = True)

evidence_short_aligned = evidence_short_aligned.dropna()

# Select evidences
group = ['CElegans_Tryp',
 'Drosophila_LysC',
 'Drosophila_LysN',
 'Drosophila_Trp',
 'Ecoli_LysC',
 'Ecoli_LysN',
 'Ecoli_trypsin',
 'HeLa_LysC',
 'HeLa_LysN',
 'HeLa_Trp_2',
 'HeLa_Trypsin_1',
 'Yeast_LysC',
 'Yeast_LysN',
 'Yeast_Trypsin']

#evidence_tmp5 = evidence_all.loc[evidence_all['Experiment'].isin(group)]


group2 = ['HeLa', 'LysN', 'LysC']
pattern = '|'.join([f"{g.strip()}" for g in group2])
group3 = ['Proteotypic', 'SRMATLAS', 'MissingGeneSet']
pattern2 = '|'.join([f"{g.strip()}" for g in group3])

evidence_tmp = evidence_all[evidence_all['Experiment'].str.contains(pattern, case=False, na=False)]
evidence_tmp2 = evidence_all[evidence_all['Experiment'].str.contains(pattern2, case=False, na=False)]

# Select evidence columns of interest
selection = ['Modified sequence', 'Sequence', 'Charge', 'Mass', 'm/z', 'Experiment', 'id',
             'Intensity', 'Score', 'Length', 'Retention time']
evidence_unique = evidence_tmp.loc[evidence_tmp.groupby(['Modified sequence', 'Charge'])['Intensity'].idxmax()][selection]

evidence_unique['CCS'] = evidence_short_aligned['CCS'].values

# Export aligned dataset for deep learning
evidence_unique.to_csv('output2/evidence_aligned_4.csv', index=False)

# Proteome Tools data
# proteometools = set(evidence_all.loc[~evidence_all['Experiment'].isin(
#     ['Proteotypic', 'SRMATLAS', 'MissingGeneSet'])]['Raw file'])

keywords = ['Proteotypic', 'SRMATLAS', 'MissingGeneSet']
pattern = '|'.join(keywords)
proteometools = set(evidence_all.loc[~evidence_all['Experiment'].str.contains(pattern, na=False)]['Raw file'])

evidence_pivot_aligned_PT = evidence_pivot_aligned.drop(proteometools, axis = 1)

# Calculate mean CCS value from aligned data
evidence_short_aligned_PT = evidence_pivot_aligned_PT.drop(columns = evidence_pivot_aligned_PT.columns.tolist())
evidence_short_aligned_PT['CCS'] = evidence_pivot_aligned_PT.mean(axis = 1, skipna = True)
evidence_short_aligned_PT.reset_index(inplace = True)
evidence_short_aligned_PT = evidence_short_aligned_PT.dropna()

# Select evidences
group = ['Proteotypic', 'SRMATLAS', 'MissingGeneSet']
pattern = '|'.join([f"{g.strip()}" for g in group])

#evidence_tmp = evidence_all.loc[evidence_all['Experiment'].isin(group)]
evidence_tmp = evidence_all[evidence_all['Experiment'].str.contains(pattern, case=False, na=False)]

# Select evidence columns of interest
selection = ['Modified sequence', 'Sequence', 'Charge', 'Mass', 'm/z', 'Experiment',
             'id', 'Intensity', 'Score', 'Length', 'Retention time']
evidence_unique_PT = evidence_tmp.loc[evidence_tmp.groupby(['Modified sequence', 'Charge'])['Intensity'].idxmax()][selection]

evidence_unique_PT['CCS'] = evidence_short_aligned_PT['CCS'].values

# Export aligned dataset for deep learning
evidence_unique_PT.to_csv('output2/evidence_aligned_PT_5.csv', index=False)



evidence_PT = evidence_pivot_aligned_PT.drop(columns = evidence_pivot_aligned_PT.columns.tolist())
evidence_PT['CCS'] = evidence_pivot_aligned_PT.mean(axis = 1, skipna = True)
evidence_PT.reset_index(inplace = True)

evidence_endo = evidence_pivot_aligned_endo.drop(columns = evidence_pivot_aligned_endo.columns.tolist())
evidence_endo['CCS'] = evidence_pivot_aligned_endo.mean(axis = 1, skipna = True)
evidence_endo.reset_index(inplace = True)

plt.scatter(evidence_endo['CCS'], evidence_PT['CCS'], s = 0.1, alpha = 0.1)
plt.xlabel('CCS ($\AA^2$), whole proteome digest')
plt.ylabel('CCS ($\AA^2$), ProteomeTools');
plt.savefig("output2/compare_6.png")
plt.close()
mask = ~np.logical_or(np.isnan(evidence_endo['CCS']), np.isnan(evidence_PT['CCS']))
pcorr = stats.pearsonr(evidence_endo['CCS'][mask], evidence_PT['CCS'][mask])[0]
print('Pearson correlation: {:1.3f}'.format(pcorr))
print('n = {0}'.format(mask.sum()))
print('Median absolute deviation = {:1.1f}%'.format(np.abs((evidence_endo['CCS'] - evidence_PT['CCS']) /
                                                           evidence_endo['CCS'] * 100).median()))