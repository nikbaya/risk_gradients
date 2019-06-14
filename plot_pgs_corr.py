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
#import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
import os

cloud_wd = 'gs://nbaya/risk_gradients/'
gwas_wd = cloud_wd+'gwas/'
phen_dict = {
    '50':['height', 360338, 0],
    '2443':['diabetes', 360142, 17272],
    '21001':['BMI',359933, 0],
}

local_wd = '/Users/nbaya/Documents/lab/risk_gradients/'


phen = '2443'
frac_all_ls = [1]
frac_cas_ls = [1]
frac_con_ls = [0.2, 0.4, 0.6, 0.8, 1]

phen = '21001'
frac_all_ls = [0.2, 0.4, 0.6, 0.8, 1]
frac_cas_ls = [1]
frac_con_ls = [1]


phen = '50'
frac_all_ls = [0.2, 0.4, 0.6, 0.8, 1]
frac_cas_ls = [1]
frac_con_ls = [1]


n = phen_dict[phen][1]
n_cas = phen_dict[phen][2]
seed = 1

df_path = f'{local_wd}{phen_dict[phen][0]}_pgs.tsv'

if os.path.isfile(df_path):
    df = pd.read_csv(df_path,delimiter='\t')
else:
    df = pd.DataFrame(columns=['phen','desc','n','n_new','n_cas','n_cas_new','r_all','pval_all','r_sub','pval_sub'])
    i = 0
    for frac_all in frac_all_ls:
        n_new0 = int(n*frac_all)
        for frac_cas in frac_cas_ls:
            n_cas_new = int(n_cas*frac_cas)
            for frac_con in frac_con_ls:
                n_new = int(frac_con*(n_new0-n_cas_new)+n_cas_new)
                if frac_con == 1 and frac_cas ==1:
                    suffix = f'.{phen}.n_{n_new}of{n}.seed_{seed}'
                else:
                    suffix = f'.{phen}.n_{n_new}of{n}.n_cas_{n_cas_new}of{n_cas}.seed_{seed}'            
                corr0 = subprocess.Popen(['gsutil','cat',f'{cloud_wd}corr{suffix}.txt'],stdout=subprocess.PIPE).communicate()[0]
                corr1 = corr0.decode('utf-8')
                if corr1 == '':
                    print(f'{cloud_wd}fcorr{suffix}.txt does not exist')
                    corr = [float('nan')]*4
                else:
                    corr2 = corr1.replace('\t','\n')
                    corr = [float(x) for x in corr2.split('\n')[:-1]]
                r_all, pval_all, r_sub, pval_sub = float(corr[0]), float(corr[1]), float(corr[2]), float(corr[3])
                df.loc[i] = [phen, phen_dict[phen][0], n, n_new, n_cas, n_cas_new, r_all, pval_all, r_sub, pval_sub]
                i += 1
    df['inv_n_new'] = 1/df.n_new
    df['inv_R2'] = 1/df.r_all**2

df = df.dropna()

plt.plot(df.n_new/df.n, df.r_all**2,'.-')
plt.xlabel('fraction of total population used in training set')
plt.ylabel('R^2')
plt.xlim([0,1])
plt.ylim([0,round(max(df.r_all**2)*1.1,2)])
plt.title(f'Prediction accuracy of PGS for {phen_dict[phen][0]} (code: {phen})')
plt.text(x=0.8,y=max(df.r_all**2)*0.05,s=f'N = {phen_dict[phen][1]}')
fig=plt.gcf()
fig.savefig(f'{local_wd}plots/{phen}_pgs_prediction.png',dpi=600)
plt.close()

if n_cas > 0:
    x = (1/df.n_cas_new + 1/(df.n_new-df.n_cas_new))
    y = 1/df.r_all**2
    plt.plot(x, y,'.-')
    plt.xlabel('1/N_e')
    plt.ylabel('1/R^2')
    result = sm.OLS(endog=y.tolist(),exog=sm.add_constant(x.tolist())).fit()
    R2 = result.rsquared
    b, a = result.params[0], result.params[1]
    plt.plot(x,a*x+b,'k--',alpha=0.5)
    plt.title(f'PGS for {phen_dict[phen][0]} (code: {phen})')
    plt.text(x=min(x), y=max(y)-0.1*(max(y)-min(y)),
             s=f'y = {round(a,6)}*x + {round(b,6)}\nR^2 = {round(R2,6)}')
    plt.legend(['1/N_e vs. 1/R^2','OLS fit'],loc='lower right')
    fig=plt.gcf()
    fig.savefig(f'{local_wd}plots/{phen}_inv_Neff_inv_R2.png',dpi=600)
    plt.close()

plt.plot(df.inv_n_new, df.inv_R2,'.-')
result = sm.OLS(df.inv_R2.tolist(),sm.add_constant(df.inv_n_new.tolist())).fit()
R2 = result.rsquared
b, a = result.params[0], result.params[1]
x = np.asarray([min(df.inv_n_new),max(df.inv_n_new)])
plt.plot(x,a*x+b,'k--',alpha=0.5)
plt.title(f'PGS for {phen_dict[phen][0]} (code: {phen})')
plt.ylabel('1/R^2')
plt.xlabel('1/N')
plt.text(x=min(df.inv_n_new), y=max(df.inv_R2)-0.2*(max(df.inv_R2)-min(df.inv_R2)),
         s=f'y = {round(a,6)}*x + {round(b,6)}\nR^2 = {round(R2,6)}\nh2_M = {round(1/b,6)}\nM_e = {int(round(a/b**2))}')
plt.legend(['1/N vs. 1/R^2','OLS fit'],loc='lower right')
fig=plt.gcf()
fig.savefig(f'{local_wd}plots/{phen}_pgs_prediction_linearized.png',dpi=600)

df.to_csv(f'{local_wd}{phen_dict[phen][0]}_pgs.tsv',sep='\t',index=False)