import pandas as pd
import random

df1 = pd.read_csv('../data_csv/evidence_aligned_4.csv')
df2 = pd.read_csv('../data_csv/evidence_aligned_PT_5.csv')

df1['PT'] = 'FALSE'
df2['PT'] = 'TRUE'

combined_df = pd.concat([df1, df2], ignore_index=True)

sorted_df = combined_df.sort_values(by='Modified sequence')

final_df = sorted_df.reset_index()[['Modified sequence', 'Charge', 'Mass', 'Intensity', 'Retention time', 'CCS', 'PT']]


final_df.reset_index(drop=True, inplace=True)

final_df.insert(0, '', range(len(final_df)))


final_df.to_csv('../data_csv/combined_ok.csv', index=False)
print('merge OK！')