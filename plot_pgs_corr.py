#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:01:30 2019

Plot results of PGS correlation

@author: nbaya
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm

cloud_wd = 'gs://nbaya/risk_gradients/'
gwas_wd = cloud_wd+'gwas/'
phen_dict = {
    '50':['height', 360338, 0],
    '2443':['diabetes', 360142, 17272],
    '21001':['bmi',359933, 0],
}

local_wd = '/Users/nbaya/Documents/lab/risk_gradients'


phen = '50'
n = phen_dict[phen][1]
n_cas = phen_dict[phen][2]
frac_all_ls = [0.2, 0.4, 0.6, 0.8, 1]
frac_cas_ls = [1]
frac_con_ls = [1]
seed = 1


df = pd.DataFrame(columns=['phen','desc','n','n_new','n_cas','n_cas_new','r_all','pval_all','r_sub','pval_sub'])


i = 0
for frac_all in frac_all_ls:
    n_new = int(n*frac_all)
    for frac_cas in frac_cas_ls:
        n_cas_new = int(n_cas*frac_cas)
        for frac_con in frac_con_ls:
            if frac_con == 1 and frac_cas ==1:
                suffix = f'.{phen}.n_{n_new}of{n}.seed_{seed}'
            else:
                suffix = '.{phen}.n_{n_new}of{n}.n_cas_{n_cas_new}of{n_cas}.seed_{seed}'            
            corr0 = subprocess.Popen(['gsutil','cat',f'{cloud_wd}corr{suffix}.txt'],stdout=subprocess.PIPE).communicate()[0]
            corr1 = corr0.decode('utf-8')
            corr2 = corr1.replace('\t','\n')
            corr = corr2.split('\n')
            r_all, pval_all, r_sub, pval_sub = float(corr[0]), float(corr[1]), float(corr[2]), float(corr[3])
            df.loc[i] = [phen, phen_dict[phen], n, n_new, n_cas, n_cas_new, r_all, pval_all, r_sub, pval_sub]
            i += 1
            
plt.plot(df.n_new/df.n, df.r_all**2,'.-')
plt.xlabel('fraction of total population used in training set')
plt.ylabel('R^2')
plt.xlim([0,1])
plt.ylim([0,0.2])
#plt.legend(['r^2 for all individuals','r^2 for subset of individuals'])
plt.title('Prediction accuracy of PGS for height')
fig=plt.gcf()
fig.savefig(f'/Users/nbaya/Downloads/{phen}_pgs_prediction.png',dpi=600)

plt.plot(1/df.n_new, 1/df.r_all**2,'.-')
#df['x'] = 1/df.n_new
#df['y'] = 1/df.r_all**2
result = sm.OLS(df.y.tolist(),sm.add_constant(df.x.tolist())).fit()
R2 = result.rsquared
slope, intercept = result.params[0], result.params[1]
plt.title('PGS for height')
plt.ylabel('1/R^2')
plt.xlabel('1/N')
fig=plt.gcf()
fig.savefig(f'/Users/nbaya/Downloads/{phen}_pgs_prediction_linearized.png',dpi=600)