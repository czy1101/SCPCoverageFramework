import tensorflow as tf

import pandas as pd

# 读取txt文件
# 请将 'your_file.txt' 替换为您的文件路径
data = pd.read_csv('../data_txt/evidence_Y.txt', sep='\t')  # 假设使用制表符分隔
data = data[data['Modified sequence'].notnull() & (data['Modified sequence'] != '')]
data = data[data['Length'] >= 7]
print(data.head())
# 提取所需列
extracted_data = data[['Modified sequence', 'Length', 'Charge', 'CCS']].copy()

# 重命名CCS列
extracted_data.rename(columns={'CCS': 'CCS_origin'}, inplace=True)

# 保存为新的csv文件
# 请将 'output_file.csv' 替换为您想要保存的文件名
extracted_data.to_csv('./predic/evidence_Y.csv', index=False)
print('ok')