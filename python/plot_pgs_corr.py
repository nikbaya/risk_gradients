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

#phi_ls = ['auto']*5+['1e-04']*5+['1e-02']*5+['unadjusted_20k.2']*5+['auto_20k.2']*5+['1e-04_20k.2']*5+['unadjusted']*5+['adjusted_for_gt']*5+['1e-04_20k.1']*5
#n_train_ls = [100e3,50e3,20e3,10e3,5e3]*9
#r_ls = ([0.01961128210830167, 0.01810725228506255, 0.012086870633101371, 0.01618182413705427, None]+
#        [0.019765671503524318, 0.01885229267730592, 0.01137816625819181, 0.015117878493456036, None]+
#        [0.013164964950006268, 0.01660796695692432, 0.0023512848724231354, 0.007431955352743687, -6.287289113239286e-05]+
#        [0.5675647978884869, 0.5130533441074888, 0.3705280797170161, 0.35187445614864826, 0.25075562987294314]+
#        [0.8000138498267761, 0.7642981905354923, 0.6261882140459507, 0.5133394741415235, 0.2814180105302948]+
#        [0.8024935965673357, None, None, None, None]+
#        [0.02342598567835493, 0.01962260208225034, 0.007687792701004736, None, None]+
#        [None, None, None, None, None]+
#        [0.7960075999310807, 0.7692464924313965, 0.8966909190525392, 0.5154029701709931, 0.3484632287132207])

phi_ls = ['1e-04.v2']*27+['pt.v2']*5+['pt.pval1e-5.v2']*5
n_train_ls = ([100e3]*3+[50e3]*6+[20e3]*6+[10e3]*6+[5e3]*6+
              [100e3,50e3,20e3,10e3,5e3]*2)
r_ls = ([0.7813541803428697, 0.7461795210965818, 0.7784047015796418, 
         0.7461795210965818, 0.7426601269397713, 0.7430362951597826, 0.7437474281984784, 0.7510667097017896, 0.7459899597930412,
         0.6326378796536457, 0.6377665057247787, 0.6273342713555827, 0.6264770581373137, 0.6288191044653261, 0.6276635627709762, 
         0.4918373903330486, 0.48183169914369894, 0.47921153243983405, 0.4757635537507058, None, None,
         0.326580942701403, 0.31876620949114404, 0.3427734970871236, 0.3228223949060413, 0.32405540671872174, 0.3072100818392767]+
        [0.41553381380344345, 0.3352062577049098, 0.23900831395025654, 0.1779981040780986, 0.1300415036081481]+
        [0.5375847672928772, 0.5374276560136735, 0.524343722267456, 0.5134197109705413, 0.47337418840176076])



df = pd.DataFrame(data=list(zip(phi_ls,n_train_ls,r_ls)),columns=['phi','n_train','r'])
df['inv_n_train'] = 1/df.n_train
df['inv_R2'] = df.r**(-2)


phi = '1e-04'
phi='auto_20k.2'
phi='1e-04_20k.1'
phi='1e-04_20k.2'
phi='unadjusted_20k.2'

phi = 'pt.v2'
phi = 'pt.pval1e-5.v2'
phi = '1e-04.v2'


df_tmp = df[df.phi==phi]
df_tmp = df_tmp.dropna(axis=0)


fig,ax = plt.subplots(figsize=(1*6,1*4))
plt.plot(df_tmp.n_train, df_tmp.r**2,'.',ms=10)
plt.xlabel('# of individuals in training set')
plt.ylabel('R^2')
plt.title(f'R^2 between PRS and simulated phenotype\nas a function of training set size (phi={phi})')
fig.savefig(f'{local_wd}plots/n_train_R2.{f"prs_cs.phi_{phi}" if "unadjusted" not in phi else f"unadjusted_betas.{phi}"}.png',dpi=600)


fig,ax = plt.subplots(figsize=(1.5*6,1.5*4))
x = (df_tmp.inv_n_train).values.reshape(-1,1)
y = (df_tmp.inv_R2).values
plt.plot(x,y,'.',ms=10)
model = LinearRegression().fit(x,y)
r2 = model.score(x,y)
b, a = model.intercept_, model.coef_[0]
plt.plot(x,a*x+b,'k--',alpha=0.5)
plt.title(f'{f"PRS-CS, phi={phi}" if "unadjusted" not in phi else f"not PRS-CS, {phi}"}')
plt.ylabel('1/R^2')
plt.xlabel('1/N')
locs, labels = plt.xticks()
plt.xticks(locs[::],[str(round(x,10)) for x in locs[::]],rotation=0)
plt.legend(['1/N vs. 1/R^2','OLS fit'],loc='lower right')
plt.text(x=min(df_tmp.inv_n_train)-(0.0)*(max(df_tmp.inv_n_train)-min(df_tmp.inv_n_train)), 
         y=max(df_tmp.inv_R2)-(0.2)*(max(df_tmp.inv_R2)-min(df_tmp.inv_R2)),
         s=f'y = {round(a,6)}*x + {round(b,6)}\nR^2 = {round(r2,6)}\nh2_M = {round(1/b,6)}\nM_e = {int(round(a/b**2))}')
fig.savefig(f'{local_wd}plots/inv_n_train_inv_R2.{f"prs_cs.phi_{phi}" if "unadjusted" not in phi else f"unadjusted_betas.{phi}"}.png',dpi=600)


#test quadratic fit
import statsmodels.formula.api as sm_api
from statsmodels.stats.anova import anova_lm

model_type = 'quad'

if model_type=='lin':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train ', data = df_tmp).fit()
elif model_type=='quad':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train + I(inv_n_train**2) ', data = df_tmp).fit()
#model = sm_api.ols(formula = 'inv_R2 ~ 1 + I(inv_n_train**2) + inv_n_train ', data = df_tmp).fit()

anova = anova_lm(model)
print(f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model')
print(model.summary())
print(anova)

fig,ax = plt.subplots(figsize=(1.5*6,1.5*4))
grouped = pd.DataFrame(data={'inv_n_train':df_tmp.groupby('inv_n_train')['inv_n_train'].mean().index,
                             'mean_inv_R2':df_tmp.groupby('inv_n_train')['inv_R2'].mean().values,
                             'std_inv_R2': df_tmp.groupby('inv_n_train')['inv_R2'].std().values})
x_obs = (df_tmp.inv_n_train).values
y_obs = (df_tmp.inv_R2).values
#p2,p1,p0 = np.polyfit(x_obs,y_obs,2) #save each coefficient as p{deg}
x = np.linspace(min(x_obs), max(x_obs),20)
y_hat = model.params.inv_n_train*x + model.params.Intercept
if  'I(inv_n_train ** 2)' in model.params:
    y_hat += model.params['I(inv_n_train ** 2)']*x**2
#plt.plot(x_obs,y_obs,'.',ms=10)
plt.errorbar(x=grouped.inv_n_train,y=grouped.mean_inv_R2,yerr=grouped.std_inv_R2*2,
             fmt='.',ms=10)
plt.plot(x,y_hat,'k--',alpha=0.5)
plt.title(f'{f"PRS-CS, phi={phi}" if "unadjusted" not in phi else f"not PRS-CS, {phi}"}')
locs, labels = plt.xticks()
plt.xticks(locs[::2],[str(round(x,10)) for x in locs[::2]],rotation=0)
plt.legend([f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model',
               '1/N vs. 1/R^2'],loc='lower right')
plt.ylabel('1/R^2')
plt.xlabel('1/N')

fig.savefig(f'{local_wd}plots/inv_n_train_inv_R2.model_{model_type}.{f"prs_cs.phi_{phi}" if "unadjusted" not in phi else f"unadjusted_betas.{phi}"}.png',dpi=600)



# read in chr22 results
df_chr22 = pd.read_csv(local_wd+'data/corr.chr22.qc_pos.maf_0.05.h2_0.75.pi_0.001.1kg_eur_hm3.phi_1e-04.tsv',
                 sep='\t')
df_chr22['inv_n_train'] = 1/df_chr22.n_train
df_chr22['inv_R2'] = df_chr22.r**(-2)

import statsmodels.formula.api as sm_api
from statsmodels.stats.anova import anova_lm

model_type = 'quad'

if model_type=='lin':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train ', data = df_chr22).fit()
elif model_type=='quad':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train + I(inv_n_train**2) ', data = df_chr22).fit()

anova = anova_lm(model)
print(f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model')
print(anova)
print(model.summary())

fig,ax = plt.subplots(figsize=(1.5*6,1.5*4))
grouped = pd.DataFrame(data={'inv_n_train':df_chr22.groupby('inv_n_train')['inv_n_train'].mean().index,
                             'mean_inv_R2':df_chr22.groupby('inv_n_train')['inv_R2'].mean().values,
                             'std_inv_R2': df_chr22.groupby('inv_n_train')['inv_R2'].std().values})
x_obs = (df_chr22.inv_n_train).values
y_obs = (df_chr22.inv_R2).values
#p2,p1,p0 = np.polyfit(x_obs,y_obs,2) #save each coefficient as p{deg}
x = np.linspace(min(x_obs), max(x_obs),20)
y_hat = model.params.inv_n_train*x + model.params.Intercept
if  'I(inv_n_train ** 2)' in model.params:
    y_hat += model.params['I(inv_n_train ** 2)']*x**2
#plt.plot(x_obs,y_obs,'.',ms=10)
plt.errorbar(x=grouped.inv_n_train,y=grouped.mean_inv_R2,yerr=grouped.std_inv_R2*2,
             fmt='.',ms=10)
plt.plot(x,y_hat,'k--',alpha=0.5)
plt.title(f'PRS-CS, chr22')
locs, labels = plt.xticks()
plt.xticks(locs[::],[str(round(x,10)) for x in locs[::]],rotation=0)
plt.legend([f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model',
               '1/N vs. 1/R^2'],loc='lower right')
plt.ylabel('1/R^2')
plt.xlabel('1/N')
fig.savefig(f'{local_wd}plots/inv_n_train_inv_R2.model_{model_type}.chr22.png',dpi=300)





# read in chr22 (unadjusted -- non-PRS-CS), h2=0.75, pi=0.001
phi_ls = ['chr22.unadjusted']*39
n_train_ls = [100e3]*3+[50e3]*6+[20e3]*10+[10e3]*10+[5e3]*10
r_ls = ([0.5260066498667789,0.5249712187300143,0.5239755562396924]+
        [0.5186951927974995,0.5228232108150805,0.5224150186253562,0.5199980317078371, 0.5275371182111939,0.5224150186253562]+
        [0.5188571467130012, 0.5105536444409511, 0.5197253689601626, 0.5202132433637073,
         0.5166925742517846, 0.5143133831414886, 0.5169984858420663, 0.5115868205889093, 0.5112739109685859, 0.5173357194453803]+
        [0.5027621749260911, 0.4989005647046562, 0.5075022530960592, 0.49821958595031385,
         0.49571834217577754, 0.4859938907715119, 0.49221941991791407,  0.501038863866561, 0.5131133195220912, 0.49769646390695527]+
        [0.4724960120788834, 0.4884036202244144, 0.479586230254173, 0.4703282418873503, 
         0.47081849511267176, 0.48272834939048903, 0.48130254617048673, 0.49065746564611723, 0.4870852805862297, 0.47444025454702576])
df = pd.DataFrame(data=list(zip(phi_ls,n_train_ls,r_ls)),columns=['phi','n_train','r'])
df['inv_n_train'] = 1/df.n_train
df['inv_R2'] = df.r**(-2)

phi = 'chr22.unadjusted'

fig,ax = plt.subplots(figsize=(1*6,1*4))
plt.plot(df.n_train, df.r**2,'.',ms=10)
plt.xlabel('# of individuals in training set')
plt.ylabel('R^2')
plt.title(f'R^2 between PRS and simulated phenotype\nas a function of training set size (phi={phi})')
fig.savefig(f'{local_wd}plots/n_train_R2.{f"prs_cs.phi_{phi}" if "unadjusted" not in phi else f"unadjusted_betas.{phi}"}.png',dpi=600)


import statsmodels.formula.api as sm_api
from statsmodels.stats.anova import anova_lm

model_type = 'lin'

if model_type=='lin':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train ', data = df).fit()
elif model_type=='quad':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train + I(inv_n_train**2) ', data = df).fit()

anova = anova_lm(model)
print(f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model')
print(anova)
#print(model.summary())

grouped = pd.DataFrame(data={'inv_n_train':df.groupby('inv_n_train')['inv_n_train'].mean().index,
                             'mean_inv_R2':df.groupby('inv_n_train')['inv_R2'].mean().values,
                             'std_inv_R2': df.groupby('inv_n_train')['inv_R2'].std().values})
x_obs = (df.inv_n_train).values
y_obs = (df.inv_R2).values
#p2,p1,p0 = np.polyfit(x_obs,y_obs,2) #save each coefficient as p{deg}
x = np.linspace(min(x_obs), max(x_obs),20)
y_hat = model.params.inv_n_train*x + model.params.Intercept
if  'I(inv_n_train ** 2)' in model.params:
    y_hat += model.params['I(inv_n_train ** 2)']*x**2
#plt.plot(x_obs,y_obs,'.',ms=10)
fig,ax = plt.subplots(figsize=(1.5*6,1.5*4))
plt.errorbar(x=grouped.inv_n_train,y=grouped.mean_inv_R2,yerr=grouped.std_inv_R2*2,
             fmt='.',ms=10)
plt.plot(x,y_hat,'k--',alpha=0.5)
plt.title(f'PRS-CS, chr22')
locs, labels = plt.xticks()
plt.xticks(locs[::],[str(round(x,10)) for x in locs[::]],rotation=0)
plt.legend([f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model',
               '1/N vs. 1/R^2'],loc='lower right')
plt.ylabel('1/R^2')
plt.xlabel('1/N')
fig.savefig(f'{local_wd}plots/inv_n_train_inv_R2.model_{model_type}.chr22.undadjusted.png',dpi=300)



## chr22, unadjusted betas, h2=0.75, pi=0.01
df = pd.read_csv(f'{local_wd}data/corr.pt.threshold_1.qc_pos.maf_0.05.h2_0.75.pi_0.01.1kg_eur_hm3.phi_None.v2.tsv',
                 sep='\t')
df['inv_n_train'] = 1/df.n_train
df['inv_R2'] = df.r**(-2)

fig,ax = plt.subplots(figsize=(1.2*6,1.2*4))
plt.plot(df.n_train, df.r**2,'.',ms=10)
plt.xlabel('# of individuals in training set')
plt.ylabel('R^2')
plt.title(f'R^2 between PRS and simulated phenotype\nas a function of training set size (unadjusted betas, chr22, h2=0.75, pi=0.01)')
fig.savefig(f'{local_wd}plots/n_train_R2.chr22.unadjusted_betas.h2_0.75.pi_0.01.png',dpi=300)



model_type = 'lin'

if model_type=='lin':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train ', data = df).fit()
elif model_type=='quad':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train + I(inv_n_train**2) ', data = df).fit()

anova = anova_lm(model)
print(f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model')
print(anova)
#print(model.summary())

grouped = pd.DataFrame(data={'inv_n_train':df.groupby('inv_n_train')['inv_n_train'].mean().index,
                             'mean_inv_R2':df.groupby('inv_n_train')['inv_R2'].mean().values,
                             'std_inv_R2': df.groupby('inv_n_train')['inv_R2'].std().values})
x_obs = (df.inv_n_train).values
y_obs = (df.inv_R2).values
#p2,p1,p0 = np.polyfit(x_obs,y_obs,2) #save each coefficient as p{deg}
x = np.linspace(min(x_obs), max(x_obs),20)
y_hat = model.params.inv_n_train*x + model.params.Intercept
if  'I(inv_n_train ** 2)' in model.params:
    y_hat += model.params['I(inv_n_train ** 2)']*x**2
#plt.plot(x_obs,y_obs,'.',ms=10)
fig,ax = plt.subplots(figsize=(1.2*6,1.2*4))
plt.errorbar(x=grouped.inv_n_train,y=grouped.mean_inv_R2,yerr=grouped.std_inv_R2*2,
             fmt='.',ms=10)
plt.plot(x,y_hat,'k--',alpha=0.5)
plt.title(f'PRS-CS, chr22')
locs, labels = plt.xticks()
plt.xticks(locs[::],[str(round(x,10)) for x in locs[::]],rotation=0)
plt.legend([f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model',
               '1/N vs. 1/R^2'],loc='lower right')
plt.ylabel('1/R^2')
plt.xlabel('1/N')
plt.title(f'Inverse R^2 between PRS and simulated phenotype\nas a function of inverse training set size (unadjusted betas, chr22, h2=0.75, pi=0.01)')
fig.savefig(f'{local_wd}plots/inv_n_train_inv_R2.chr22.unadjusted_betas.h2_0.75.pi_0.01.png',dpi=300)



# chr22 PRS-CS, h2=0.75, pi=1

df = pd.read_csv(local_wd+'data/corr.qc_pos.maf_0.05.h2_0.75.pi_1.1kg_eur_hm3.phi_1e-04.chrom_22.v2.tsv',
                 sep='\t')

fig,ax = plt.subplots(figsize=(1.2*6,1.2*4))
plt.plot(df.n_train, df.r**2,'.',ms=10)
plt.xlabel('# of individuals in training set')
plt.ylabel('R^2')
plt.title(f'R^2 between PRS and simulated phenotype\nas a function of training set size (PRS-CS, chr22, h2=0.75, pi=1)')
fig.savefig(f'{local_wd}plots/n_train_R2.chr22.prs_cs.phi_1e-04.h2_0.75.pi_1.png',dpi=300)


df['inv_n_train'] = 1/df.n_train
df['inv_R2'] = df.r**(-2)

import statsmodels.formula.api as sm_api
from statsmodels.stats.anova import anova_lm

model_type = 'lin'

if model_type=='lin':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train ', data = df).fit()
elif model_type=='quad':
    model = sm_api.ols(formula = 'inv_R2 ~ 1 + inv_n_train + I(inv_n_train**2) ', data = df).fit()

anova = anova_lm(model)
print(f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model')
print(anova)
#print(model.summary())

grouped = pd.DataFrame(data={'inv_n_train':df.groupby('inv_n_train')['inv_n_train'].mean().index,
                             'mean_inv_R2':df.groupby('inv_n_train')['inv_R2'].mean().values,
                             'std_inv_R2': df.groupby('inv_n_train')['inv_R2'].std().values})
x_obs = (df.inv_n_train).values
y_obs = (df.inv_R2).values
#p2,p1,p0 = np.polyfit(x_obs,y_obs,2) #save each coefficient as p{deg}
x = np.linspace(min(x_obs), max(x_obs),20)
y_hat = model.params.inv_n_train*x + model.params.Intercept
if  'I(inv_n_train ** 2)' in model.params:
    y_hat += model.params['I(inv_n_train ** 2)']*x**2
#plt.plot(x_obs,y_obs,'.',ms=10)
fig,ax = plt.subplots(figsize=(1.2*6,1.2*4))
plt.errorbar(x=grouped.inv_n_train,y=grouped.mean_inv_R2,yerr=grouped.std_inv_R2*2,
             fmt='.',ms=10)
plt.plot(x,y_hat,'k--',alpha=0.5)
plt.title(f'PRS-CS, chr22')
locs, labels = plt.xticks()
plt.xticks(locs[::],[str(round(x,10)) for x in locs[::]],rotation=0)
plt.legend([f'{("Linear" if model_type is "lin" else ("Quadratic" if model_type is "quad" else "Unspecified"))} model',
               '1/N vs. 1/R^2'],loc='lower right')
plt.ylabel('1/R^2')
plt.xlabel('1/N')
plt.title(f'Inverse R^2 between PRS and simulated phenotype\nas a function of inverse training set size (PRS-CS, chr22, h2=0.75, pi=1)')
fig.savefig(f'{local_wd}plots/inv_n_train_inv_R2.chr22.prs_cs.phi_1e-04.h2_0.75.pi_1.png',dpi=300)




df = pd.read_csv('/Users/nbaya/Documents/lab/risk_gradients/data/genetic_map_chr1_combined_b37.txt',
                 delim_whitespace=True)
df = df.rename(columns={'COMBINED_rate(cM/Mb)':'rate'})
rec_rate_thresh = 50
peak_radius = 500000 # radius in base pairs around a local peak in recombination rate
max_ldblk_len = peak_radius*10 # maximum length of an LD block in base pairs
hotspot_idx = [1758, 2575, 5155, 7170, 8527, 11131, 11483, 13281, 14332, 17449, 20437, 22337, 24575, 27431, 27937, 29128, 31319, 32379, 33252, 33889, 34677, 36076, 36455, 37395, 41123, 44278, 46088, 50876, 51691, 53637, 55089, 56365, 57665, 62467, 65353, 66764, 71744, 75911, 76883, 79257, 79993, 83874, 86524, 87976, 89389, 91070, 92415, 93520, 95853, 96478, 100319, 102392, 104411, 105519, 108481, 109629, 116434, 117094, 118098, 118607, 122249, 124240, 126565, 127989, 131556, 133297, 136159, 136161, 139509, 140574, 142655, 143887, 144548, 146154, 148654, 150100, 151328, 152664, 153614, 155020, 156791, 158867, 160605, 162991, 164526, 165375, 168042, 170041, 172518, 174987, 176113, 177109, 178546, 181691, 182664, 184002, 184851, 185822, 187023, 189044, 193789, 194718, 195463, 197772, 199372, 201334, 202956, 208709, 209850, 212921, 213629, 215377, 217522, 218403, 223938, 224028, 225742, 227052, 228296, 230715, 233027, 234499, 235372, 236365, 237452, 238460, 239450, 240661, 241332, 243534, 244495, 245816, 248198]

plt.plot(df.position, df.rate)
plt.axhline(y=50, ls='--', alpha=0.5, c = 'k')
plt.errorbar(df.loc[hotspot_idx].position, [80+ np.random.normal(scale=2) for _ in hotspot_idx],xerr =peak_radius, 
             fmt='r.', elinewidth=2)
plt.xlim([0, 2e7])
plt.xlabel('position (base pairs)')
plt.ylabel('Recombination rate (cM/Mb)')
plt.savefig('/Users/nbaya/Downloads/ldblocks.png',dpi=300)