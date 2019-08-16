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
from sklearn.linear_model import LinearRegression
import os

cloud_wd = 'gs://nbaya/risk_gradients/'
gwas_wd = cloud_wd+'gwas/'
phen_dict = {
    '50':['height', 360338, 0, 324304, 0],
    '2443':['diabetes', 360142, 17272, 324128, 15494],
    '21001':['BMI',359933, 0, 323940, 0]
}


local_wd = '/Users/nbaya/Documents/lab/risk_gradients/'


phen = '2443'
frac_all_ls = [1]
frac_cas_ls = [0.25, 0.5, 0.75, 1]
frac_con_ls = [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1]

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
n_train = phen_dict[phen][3]
n_cas_train = phen_dict[phen][4]
seed = 1
thresholds = [1]

#df_path = f'{local_wd}data/pgs_results.{phen}.{phen_dict[phen][0]}.tsv' #PGS from real phenotypes
df_path = f'corr.qc_pos.maf_0.05.h2_0.75.pi_0.001.1kg_eur_hm3.tsv' #PGS from simulated phenotype (h2=0.75, pi=0.001)

if os.path.isfile(df_path):
    df = pd.read_csv(df_path,delimiter='\t')
else:
    df = pd.DataFrame(columns=['phen','desc','n','n_cas','n_train','n_cas_train',
                               'n_test','n_cas_test','threshold','r','pval'])
    i = 0
    for frac_all in frac_all_ls:
        n_new0 = int(n_train*frac_all)
        for frac_cas in frac_cas_ls:
            n_cas_new = int(n_cas_train*frac_cas)
            n_new1 = (n_new0 - n_cas_train) + n_cas_new
            for frac_con in frac_con_ls:
                n_new = int(frac_con*(n_new1-n_cas_new)+n_cas_new)
                for t in thresholds:
                    if frac_con == 1 and frac_cas ==1:
                        suffix = f'.{phen}.n_{n_new}of{n_train}.seed_{seed}.threshold_{t}'
                    else:
                        suffix = f'.{phen}.n_{n_new}of{n_train}.n_cas_{n_cas_new}of{n_cas_train}.seed_{seed}.threshold_{t}'
                    if not os.path.isfile(f'{local_wd}data/corr{suffix}.tsv'):
                        subprocess.call(f'gsutil cp {cloud_wd}corr{suffix}.tsv {local_wd}data/'.split(' '))
                    if not os.path.isfile(f'{local_wd}data/corr{suffix}.tsv'):
                        print(f'{cloud_wd}corr{suffix}.tsv does not exist, using null values instead')
                        row = [float('nan')]*11
                    else:
                        result = pd.read_csv(f'{local_wd}data/corr{suffix}.tsv',sep='\t')
                        row = [result['phen'].values[0],result['desc'].values[0],n,n_cas,n_new,n_cas_new,
                                     result['n_test'].values[0],result['n_cas_test'].values[0],
                                     t, result['r'].values[0], result['pval'].values[0]]
                    df.loc[i] = row
                i += 1
    df['inv_n_train'] = 1/df.n_train
    df['inv_R2'] = 1/df.r**2

df = df.dropna()


df = df.sort_index()
for frac in frac_cas_ls:
    df_tmp = df[df.n_cas_train == int(phen_dict[phen][4]*frac)]    
    plt.plot(df_tmp.n_train/phen_dict[phen][3], df_tmp.r**2,'.-')
plt.xlabel('fraction of training set used for GWAS')
plt.ylabel('R^2')
plt.xlim([0,1])
plt.ylim([0,round(max(df.r**2)*1.1,(2 if max(df.r**2)>0.1 else 3))])
plt.title(f'Prediction accuracy of PGS for {phen_dict[phen][0]} (code: {phen})')
plt.text(x=(0.75 if n_cas>0 else 0.7),
         y=max(df.r**2)*(0.06 if n_cas>0 else 0.06),
         s=(f'N_train = {phen_dict[phen][3]}'+(f'\nN_cas_train = {phen_dict[phen][4]}' if n_cas>0 else '')+
                         f'\nN_test = {n-phen_dict[phen][3]}'+(f'\nN_cas_test = {n_cas-phen_dict[phen][4]}' if n_cas>0 else '')),
         fontsize=8)
if len(frac_cas_ls)>1:
    plt.legend([f'fraction of n_cas_train: {frac}' for frac in frac_cas_ls])
fig=plt.gcf()
fig.set_size_inches(6*1.2,4*1.2)
fig.savefig(f'{local_wd}plots/{phen}_pgs_prediction.png',dpi=600)
plt.close()

if n_cas > 0:
    df = df.sort_values(by='r')
    x = np.asarray(1/df.n_cas_train + 1/(df.n_train-df.n_cas_train)).reshape((-1,1))
    y = np.asarray(1/df.r**2)
    plt.plot(x, y,'.',ms=10)
    plt.xlabel('1/N_e')
    plt.ylabel('1/R^2')
    model = LinearRegression().fit(x,y)
    r2 = model.score(x,y)
    b, a = model.intercept_, model.coef_[0]
    plt.plot(x,a*x+b,'k--',alpha=0.5)
    plt.title(f'PGS for {phen_dict[phen][0]} (code: {phen})')
    plt.text(x=min(x), y=max(y)-0.1*(max(y)-min(y)),
             s=f'y = {round(a,6)}*x + {round(b,6)}\nR^2 = {round(r2,6)}')
    plt.legend(['1/N_e vs. 1/R^2','OLS fit'],loc='lower right')
    locs, labels = plt.xticks()
    plt.xticks(locs[::2],[str(round(x,10)) for x in locs[::2]],rotation=0)
    fig=plt.gcf()
    fig.set_size_inches(6*1.2, 4*1.2)
    fig.savefig(f'{local_wd}plots/{phen}_inv_Neff_inv_R2.png',dpi=600)
    plt.close()

for i, frac in enumerate(frac_cas_ls):
    df_tmp = df[df.n_cas_train == int(phen_dict[phen][4]*frac)].sort_values(by='n_train')
    x = np.asarray(df_tmp.inv_n_train).reshape((-1,1))
    y = np.asarray(df_tmp.inv_R2)
    plt.plot(x, y,'.-',c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
    
    model = LinearRegression().fit(x,y)
    r2 = model.score(x,y)
    b, a = model.intercept_, model.coef_[0]
    plt.plot(x,a*x+b,'k--',alpha=0.5)
    plt.title(f'PGS for {phen_dict[phen][0]} (code: {phen})')
    plt.ylabel('1/R^2')
    plt.xlabel('1/N')
    locs, labels = plt.xticks()
    plt.xticks(locs[::2],[str(round(x,10)) for x in locs[::2]],rotation=0)
    plt.legend(['1/N vs. 1/R^2','OLS fit'],loc='lower right')
    plt.text(x=min(df_tmp.inv_n_train), 
             y=max(df_tmp.inv_R2)-(0.2 if len(frac_cas_ls)==1 else 0.3)*(max(df_tmp.inv_R2)-min(df_tmp.inv_R2)),
             s=f'y = {round(a,6)}*x + {round(b,6)}\nR^2 = {round(r2,6)}\nh2_M = {round(1/b,6)}\nM_e = {int(round(a/b**2))}'+
             f'\nfrac_cas = {frac}' if len(frac_cas_ls)>1 else '')
    fig=plt.gcf()
    fig.savefig(f'{local_wd}plots/{phen}_pgs_prediction_linearized{"" if len(frac_cas_ls)==1 else f".frac_cas_{frac}"}.png',dpi=600)
#    fig.savefig(f'{local_wd}plots/{phen}_pgs_prediction_linearized_all.png',dpi=600) #only use if len(frac_cas_ls) > 1
    plt.close()


df.to_csv(df_path,sep='\t',index=False)



# manually create df of correlation results for PRS-CS on simulated phenotype

phi_ls = ['auto']*5+['1e-04']*5+['1e-02']*5+['unadjusted_20k.2']*5+['auto_20k.2']*5+['1e-04_20k.2']*5+['unadjusted']*5+['adjusted_for_gt']*5+['1e-04_20k.1']*5
n_train_ls = [100e3,50e3,20e3,10e3,5e3]*9
r_ls = ([0.01961128210830167, 0.01810725228506255, 0.012086870633101371, 0.01618182413705427, None]+
        [0.019765671503524318, 0.01885229267730592, 0.01137816625819181, 0.015117878493456036, None]+
        [0.013164964950006268, 0.01660796695692432, 0.0023512848724231354, 0.007431955352743687, -6.287289113239286e-05]+
        [0.5675647978884869, 0.5130533441074888, 0.3705280797170161, 0.35187445614864826, 0.25075562987294314]+
        [0.8000138498267761, 0.7642981905354923, 0.6261882140459507, 0.5133394741415235, None]+
        [0.8024935965673357, None, None, None, None]+
        [0.02342598567835493, 0.01962260208225034, 0.007687792701004736, None, None]+
        [None, None, None, None, None]+
        [None, None, None, 0.5154029701709931, None])

df = pd.DataFrame(data=list(zip(phi_ls,n_train_ls,r_ls)),columns=['phi','n_train','r'])
df['inv_n_train'] = 1/df.n_train
df['inv_R2'] = df.r**(-2)


phi = '1e-04'
phi='unadjusted_20k.2'
phi='auto_20k'


df_tmp = df[df.phi==phi]
df_tmp = df_tmp.dropna(axis=0)

x = (df_tmp.inv_n_train).values.reshape(-1,1)
y = (df_tmp.inv_R2).values

fig,ax = plt.subplots(figsize=(1.5*6,1.5*4))
plt.plot(x,y,'.',ms=10)
model = LinearRegression().fit(x,y)
r2 = model.score(x,y)
b, a = model.intercept_, model.coef_[0]
plt.plot(x,a*x+b,'k--',alpha=0.5)
plt.title(f'{f"PRS-CS, phi={phi}" if "unadjusted" not in phi else f"not PRS-CS, {phi}"}')
plt.ylabel('1/R^2')
plt.xlabel('1/N')
locs, labels = plt.xticks()
plt.xticks(locs[::2],[str(round(x,10)) for x in locs[::2]],rotation=0)
plt.legend(['1/N vs. 1/R^2','OLS fit'],loc='lower right')
plt.text(x=min(df_tmp.inv_n_train)-(0.0)*(max(df_tmp.inv_n_train)-min(df_tmp.inv_n_train)), 
         y=max(df_tmp.inv_R2)-(0.2)*(max(df_tmp.inv_R2)-min(df_tmp.inv_R2)),
         s=f'y = {round(a,6)}*x + {round(b,6)}\nR^2 = {round(r2,6)}\nh2_M = {round(1/b,6)}\nM_e = {int(round(a/b**2))}')
fig.savefig(f'{local_wd}plots/{f"prs_cs.phi_{phi}" if "unadjusted" not in phi else f"unadjusted_betas.{phi}"}.png',dpi=600)
