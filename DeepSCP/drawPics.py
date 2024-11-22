import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
def calculate_cv_ignore_zeros(matrix):
    means = []
    stds = []
    num_nonzero_columns = []

    for row in matrix:
        # non_zero_values = row[row != 0]
        # print('94',row)
        #non_zero_values = row.drop('Leading razor protein', errors='ignore')

        non_zero_values = row[row != 0]
        if len(non_zero_values) > 0:
            #print(non_zero_values)
            mean = np.mean(non_zero_values)
            std = np.std(non_zero_values, ddof=1)
            means.append(mean)
            stds.append(std)
            num_nonzero_columns.append(len(non_zero_values))
        else:
            means.append(0)
            stds.append(0)
            num_nonzero_columns.append(0)
    # print(stds)
    # print(means)

    cv = np.array(stds) / np.array(means)
    cv[np.array(num_nonzero_columns) == 0] = 0  
    return cv
#from sklearn.feature_selection import mutual_info_classif

# #nano
# file_path = '/home/deepCCS/DeepCollisionalCrossSection-train/predic/dfSP_nano_mamba_pred.csv'#有mamba
# dfdb = pd.read_csv(file_path)
# df_selected = dfdb[['Sequence','label','DeltaRT', 'PEPRT', 'ScoreRT','Cosine','PEPCosine','ScoreCosine','CCS_prediction']]
# df_selected.to_csv('./pics/df_selected_nano.csv', index=False)
#
# #N2
# file_path = '/home/deepCCS/DeepCollisionalCrossSection-train/predic/dfSP_N2_mamba_pred.csv'
# dfdb = pd.read_csv(file_path)
# df_selected = dfdb[['Sequence','label','DeltaRT', 'PEPRT', 'ScoreRT','Cosine','PEPCosine','ScoreCosine','CCS_prediction']]
# df_selected.to_csv('./pics/df_selected_N2.csv', index=False)
#
#house
# file_path='/home/deepCCS/DeepCollisionalCrossSection-train/predic/dfSP_houseMamba_part0_pred.csv'
# dfdb3 = pd.read_csv(file_path)
# file_path2='/home/deepCCS/DeepCollisionalCrossSection-train/predic/dfSP_houseMamba_part1_pred.csv'
# dfdb2 = pd.read_csv(file_path2)
# dfdb= pd.concat([dfdb3, dfdb2], axis=0, ignore_index=True)
# df_selected = dfdb[['Sequence','label','DeltaRT', 'PEPRT', 'ScoreRT','Cosine','PEPCosine','ScoreCosine','CCS_prediction']]
# #df_selected.to_csv('./pics/df_selected_house.csv', index=False)

#例子
#dfRT= pd.read_csv('./example/data/inHouse/dfSP_czy.csv', sep=',', low_memory=False)
# df_selected = dfRT[['label','DeltaRT', 'PEPRT', 'ScoreRT','Cosine','PEPCosine','ScoreCosine']]
# df_selected.to_csv('./pics/df_selected.csv', index=False)


dfRT = pd.read_csv('./pics/df_selected_nano.csv', sep=',', low_memory=False)
#dfRT = dfRT.drop_duplicates(subset='Sequence')

cvdata = pd.read_csv('./pics/DeepSCP_pro_fthouse.csv', sep=',', low_memory=False)







flag =False
flag2 =True
#RT
if flag:
    # region RT阶段画图

    targets_Del = dfRT[dfRT['label'] == 1]['DeltaRT']
    decoys_Del = dfRT[dfRT['label'] == 0]['DeltaRT']
    # 使用Seaborn绘制核密度估计图
    sns.kdeplot(targets_Del, label='targets', fill=True)
    sns.kdeplot(decoys_Del, label='decoys', fill=True)
    #plt.xlim(-1, 6)
    plt.legend()
    plt.xlabel('DeltaRT')
    plt.ylabel('Density')
    plt.title('Density Plot of Del Data')
    #plt.savefig('./pics/DeltaRTdensity_plot.pdf', format='pdf')
    plt.savefig('./pics/DeltaRTdensity_plot.png', format='png')

    # 清除当前图形内容
    plt.clf()

    targets_PEP = dfRT[dfRT['label'] == 1]['PEPRT']
    decoys_PEP = dfRT[dfRT['label'] == 0]['PEPRT']
    # 使用Seaborn绘制核密度估计图
    sns.kdeplot(targets_PEP, label='targets', fill=True)
    sns.kdeplot(decoys_PEP, label='decoys', fill=True)
    plt.xlim(right=50)  # 只限制最大值，最小值自动调整
    plt.xlim(-10, 55)

    plt.legend()
    plt.xlabel('PEPRT')
    plt.ylabel('Density')
    plt.title('Density Plot of PEPRT Data')
    #plt.savefig('./pics/PEPRTdensity_plot.pdf', format='pdf')
    plt.savefig('./pics/PEPRTdensity_plot.png', format='png')

    # 清除当前图形内容
    plt.clf()

    targets_score = dfRT[dfRT['label'] == 1]['ScoreRT']
    decoys_score = dfRT[dfRT['label'] == 0]['ScoreRT']
    # 使用Seaborn绘制核密度估计图
    sns.kdeplot(targets_score, label='targets', fill=True)
    sns.kdeplot(decoys_score, label='decoys', fill=True)
    plt.xlim(right=150)  # 只限制最大值，最小值自动调整


    plt.legend()
    plt.xlabel('ScoreRT')
    plt.ylabel('Density')
    plt.title('Density Plot of ScoreRT Data')
    #plt.savefig('./pics/ScoreRTdensity_plot.pdf', format='pdf')
    plt.savefig('./pics/ScoreRTdensity_plot.png', format='png')

    # endregion

#Cosine
if flag:
 
    targets_Cosine = dfRT[dfRT['label'] == 1]['Cosine']
    decoys_Cosine = dfRT[dfRT['label'] == 0]['Cosine']
 
    sns.kdeplot(targets_Cosine, label='targets', fill=True)
    sns.kdeplot(decoys_Cosine, label='decoys', fill=True)

    plt.legend()
    plt.xlabel('Cosine')
    plt.ylabel('Density')
    plt.title('Density Plot of Cosine Data')
    #plt.savefig('./pics/Cosinedensity_plot.pdf', format='pdf')
    plt.savefig('./pics/Cosinedensity_plot.png', format='png')

 
    plt.clf()

    targets_PEPCosine = dfRT[dfRT['label'] == 1]['PEPCosine']
    decoys_PEPCosine = dfRT[dfRT['label'] == 0]['PEPCosine']
 
    sns.kdeplot(targets_PEPCosine, label='targets', fill=True)
    sns.kdeplot(decoys_PEPCosine, label='decoys', fill=True)

    plt.legend()
    plt.xlabel('PEPCosine')
    plt.ylabel('Density')
    plt.title('Density Plot of PEPCosine Data')
    #plt.savefig('./pics/PEPCosinedensity_plot.pdf', format='pdf')
    plt.savefig('./pics/PEPCosinedensity_plot.png', format='png')

 
    plt.clf()

    targets_ScoreCosine = dfRT[dfRT['label'] == 1]['ScoreCosine']
    decoys_ScoreCosine = dfRT[dfRT['label'] == 0]['ScoreCosine']
 
    sns.kdeplot(targets_ScoreCosine, label='targets', fill=True)
    sns.kdeplot(decoys_ScoreCosine, label='decoys', fill=True)
    plt.xlim(right=150)   

    plt.legend()
    plt.xlabel('ScoreCosine')
    plt.ylabel('Density')
    plt.title('Density Plot of ScoreCosine Data')
    #plt.savefig('./pics/ScoreCosinedensity_plot.pdf', format='pdf')
    plt.savefig('./pics/ScoreCosinedensity_plot.png', format='png')

    #endregion

#CCS
if flag:
 
    targets_Cosine = dfRT[dfRT['label'] == 1]['CCS_prediction']
    targets_Cosine_normalized = (targets_Cosine - targets_Cosine.min()) / (targets_Cosine.max() - targets_Cosine.min())
    targets_Cosine=targets_Cosine_normalized
    #targets_Cosine= np.log(dfRT[dfRT['label'] == 1]['CCS_prediction']  )
    decoys_Cosine = dfRT[dfRT['label'] == 0]['CCS_prediction']
    decoys_Cosine_normalized = (decoys_Cosine - decoys_Cosine.min()) / (decoys_Cosine.max() - decoys_Cosine.min())
    print(decoys_Cosine_normalized)
    decoys_Cosine=decoys_Cosine_normalized
    #decoys_Cosine =np.log(dfRT[dfRT['label'] == 0]['CCS_prediction'] )
    sns.kdeplot(targets_Cosine, label='targets', fill=True)
    sns.kdeplot(decoys_Cosine, label='decoys', fill=True)
    #plt.xlim(200, 1100)

    plt.legend()
    plt.xlabel('CCS_prediction')
    plt.ylabel('Density')
    plt.title('Density Plot of CCS Data')
    #plt.savefig('./pics/Cosinedensity_plot.pdf', format='pdf')
    plt.savefig('./pics/CCS_plot.png', format='png')


    #endregion

#CV
if flag:
    cvdata = cvdata.apply(pd.to_numeric, errors='coerce')
    matrix = cvdata.to_numpy()
    cv = calculate_cv_ignore_zeros(matrix)

    cvdata['CV'] = cv


    plt.figure(figsize=(6, 4))

    sns.kdeplot(data=cvdata, x="CV", fill=True)

    plt.xlabel("CV")
    plt.ylabel("Density")
    plt.title("CV Density for Different Tools")
    plt.savefig('./pics/cv_plot.pdf', format='pdf')

#umap
if flag:


    cvdata = cvdata.drop(columns=['Leading razor protein'])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)


    embedding = reducer.fit_transform(cvdata)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c='skyblue', s=30, cmap='Spectral')
    plt.title('UMAP Visualization of Protein Data')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.colorbar()
    plt.savefig('./pics/cv_plot.pdf', format='pdf')
    # endregion

print('ok')