import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import seaborn as sns

dic = {}

for i in tqdm(os.listdir('/home/nima/OpenOOD/results')):
    if ('test_ood' in i):
        method = i.split('_')[6]
        csv = pd.read_csv('./results/'+i+'/ood.csv')
        dic[method] = np.array(csv['FPR@95'])
dic['dataset'] = csv['dataset']
print(pd.DataFrame(dic))
df = pd.DataFrame(dic)
df = pd.melt(df, id_vars='dataset', var_name='method', value_name='fpr@95')
df.to_csv('./scripts/classifier_2/fpr95comparison.csv')
plot = sns.barplot(x='dataset', y='fpr@95', hue='method', data=df)
plot.get_figure().savefig('scripts/classifier_2/fpr.png')