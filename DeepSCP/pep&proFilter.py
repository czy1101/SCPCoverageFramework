import pandas as pd

# 加载蛋白质组学数据
df = pd.read_csv('peptides.txt', sep='\t')

# 筛选高质量蛋白质
filtered_df = df[
    #(df['Unique peptides'] > 2) &                       # 筛选独特肽段数量大于2
    #(df['Score'] > 20) &
    #(df['Q-value'] <0.01) & #筛选蛋白质得分大于30
    #(df['Sequence coverage [%]'] > 20) &               # 筛选序列覆盖率大于20%

    (df['Length'] <= 47) &                       # 长度至少为 7（常见的肽段长度标准）
    (df['PEP'] < 0.01) &                        # PEP 值小于 0.01

    #(df['Score'] > 30) &                        # Score 值大于 30（根据你的数据选择合适的阈值）
    #(df['Missed cleavages'] <= 2) &

    (df['Reverse'] != '+') &                           # 排除 Reverse 蛋白
    (df['Potential contaminant'] != '+')               # 排除 Potential contaminant 蛋白
]


# print(df['sequence coverage [%]'])
# print(df['Q-value	Score'])

#print(df.columns.tolist())
print(len(filtered_df))