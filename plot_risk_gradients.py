#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:06:45 2019

@author: nbaya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #; sns.set(color_codes=False)
import scipy.stats as stats

df = pd.read_csv('/Users/nbaya/Documents/lab/risk_gradients/gps_6dis_alex_4.25.19.csv.gz',compression='gzip')
df = df.rename(index=str,columns={'bmi1':'bmi'})
no_nan = df.loc[False==np.isnan(df.gpsbmi)].copy()  #remove NaNs from gpsbmi to ensure bivariate plot for BMI works

"""
field descriptions:
    eid (int): individual ID
    age (int): age
    sex (bool): 0 corresponds with female, 1 corresponds with male (based on assumptions from gpscad, breastcanc) 
    genotyping_array (str) : options={UKBB: 285370 of 288940 (~98.76%), UKBL: 3570 of 288940 (~1.236%)}
    PCs (float: 1 to 10
    gpscad (float): genome-wide polygenic score for coronary artery disease
    cad (bool): diagnosed with CAD?
    gpsbreastcanc (float): genome-wide polygenic score for breast cancer
    breastcanc (bool): diagnosed with breast cancer?
    gpsaf (float): genome-wide polygenic score of atrial fibrillation
    af (bool): diagnosed with atrial fibrillation
    gpsdm (float): genome-wide polygenic score of diabetes mellitus
    dm (bool): diagnosed with diabetes mellitus
    gpsibd (float): genome-wide polygenic score of inflammatory bowel disease
    ibd (bool): diagnosed with inflammatory bowel disease
    gpsbmi (float): genome-wide polygenic score of body mass index
    bmi (float): body mass index
    sevobese (bool): is severely obese
""" 
diseases = {'cad':'CAD','af':'atrial fibrillation','dm':'type 2 diabetes',
            'ibd':'inflammatory bowel disease','breastcanc':'breast cancer',
            'bmi':'body mass index'}


def risk_gradient(df, disease, x_axis='percentile',y_axis='avg_prev'):
    """
    Options for x_axis: percentile, avg_gps
                y_axis: avg_prev, logit, inv_cdf, avg
    """
    xlabel_dict = {'percentile': 'Percentile of polygenic score',
                   'avg_gps': 'Avg. GPS of percentile bin'}
    ylabel_dict = {'avg_prev':f'Prevalence of {diseases[disease]} (%)',
                   'logit': 'log(K/(1-K))',
                   'inv_cdf':'Phi^-1(K)',
                   'avg':f'Average {diseases[disease]}'}
    rank = stats.rankdata(df['gps'+disease])
    percentile = rank/len(df)*100
    
    bins = list(range(0,100))
    bin_idx = np.digitize(percentile, bins, right=True)
    
    df['bin_idx'] = bin_idx
    
    avg = np.asarray([np.mean(df[df.bin_idx==i][disease]) for i in range(1,101)]) #average of disease in percentile bin
    avg_prev = 100*avg
    avg_gps = [np.mean(df[df.bin_idx==i]['gps'+disease]) for i in range(1,101)]
    logit = [np.log(k/(1-k)) for k in avg_prev]
    inv_cdf = [stats.norm.ppf(k) for k in avg_prev]
    
    fig,ax=plt.subplots(figsize=(4.5,5))
    if x_axis == 'percentile': #no transformation, x-axis is percentile of bins, y-axis is prevalence as %
        if disease=='bmi':
            y_axis='avg'
            plt.scatter(range(1,101),avg,c=range(1,101),cmap='Blues',vmin=-130,vmax=100,s=20)    
        else:
            plt.scatter(range(1,101),avg_prev,c=range(1,101),cmap='Blues',vmin=-130,vmax=100,s=20)    
        plt.xticks(np.linspace(0,100,11))
    elif disease == 'bmi' and x_axis == 'avg_gps': #if disease is bmi and we transform x-axis to be the avg gps for each percentile bin        
        y_axis = 'avg'
        plt.scatter(avg_gps,avg_prev,c=range(1,101),cmap='Blues',vmin=-130,vmax=100,s=20)    
    elif x_axis=='avg_gps' and y_axis=='logit': #x-axis is avg gps for each percentile bin, y-axis is logit transformation of prevalence
        plt.scatter(avg_gps,logit,c=range(1,101),cmap='Blues',vmin=-130,vmax=100,s=20)    
    elif x_axis=='avg_gps' and y_axis=='inv_cdf': #x-axis is avg gps for each percentile bin, y-axis is inverse cdf transformation of prevalence
        plt.scatter(avg_gps,inv_cdf,c=range(1,101),cmap='Blues',vmin=-130,vmax=100,s=20)    
    else:
        return None
    plt.xlabel(xlabel_dict[x_axis])
    plt.ylabel(ylabel_dict[y_axis])
    plt.title(f'Risk gradient for {diseases[disease]}')
    plt.tight_layout()
    fig=plt.gcf()
    fig.savefig(f'/Users/nbaya/Documents/lab/risk_gradients/plots/{disease}_riskgradient_xaxis{x_axis}_yaxis{y_axis}.png',dpi=600)
    plt.close()


for disease, fullname in diseases.items():
    x_axis, y_axis = 'percentile','avg_gps'
    risk_gradient(df, disease,x_axis=x_axis,y_axis=y_axis)
    x_axis='avg_gps'
    for y_axis in ['avg_gps','logit','inv_cdf']:
        risk_gradient(df, disease,x_axis=x_axis,y_axis=y_axis)

# Plotting for sevobese
tmp = no_nan.copy() #use version of dataset with no NaNs in gpsbmi
rank = stats.rankdata(tmp['gpsbmi'])
percentile = rank/len(tmp)*100
bins = list(range(0,100))
bin_idx = np.digitize(percentile, bins, right=True)
tmp['bin_idx'] = bin_idx
avg_prev = np.asarray([np.mean(tmp[tmp.bin_idx==i]['sevobese']) for i in range(1,101)]) #average prevalence of disease in percentile bin
avg_gps = [np.mean(tmp[tmp.bin_idx==i]['gpsbmi']) for i in range(1,101)]
logit = [np.log(k/(1-k)) for k in avg_prev]
inv_cdf = [stats.norm.ppf(k) for k in avg_prev]

fig,ax=plt.subplots(figsize=(4.5,5))
plt.scatter(range(1,101),avg_prev*100,c=range(1,101),cmap='Blues',vmin=-130,vmax=100,s=20)    
plt.xlabel('Percentile of polygenic score for BMI')
plt.ylabel('Prevalence of severe obesity (%)')
plt.title(f'Risk gradient for severe obesity')
plt.tight_layout()
fig=plt.gcf()
fig.savefig(f'/Users/nbaya/Documents/lab/risk_gradients/plots/sevobese_riskgradient_xaxisavg_gps_yaxisavg_prev.png',dpi=600)

ylabel_dict = {'logit': 'log(K/(1-K))',
               'inv_cdf':'Phi^-1(K)'}
for y_axis, ylabel in ylabel_dict.items():
    fig,ax=plt.subplots(figsize=(4.5,5))
    if y_axis=='logit':
        plt.scatter(avg_gps,logit,c=range(1,101),cmap='Blues',vmin=-130,vmax=100,s=20)    
    else:
        plt.scatter(avg_gps,inv_cdf,c=range(1,101),cmap='Blues',vmin=-130,vmax=100,s=20)    
    plt.ylabel(ylabel)
    plt.xlabel('Avg. GPS of percentile bin for BMI')
    plt.title(f'Risk gradient for {diseases[disease]}')
    plt.tight_layout()
    fig=plt.gcf()
    fig.savefig(f'/Users/nbaya/Documents/lab/risk_gradients/plots/sevobese_riskgradient_xaxisavg_gps_yaxisavg_prev.png',dpi=600)




# Plotting bivariate distribution for BMI
R,p = stats.pearsonr(no_nan.bmi,no_nan.gpsbmi)

fig,ax = plt.subplots(figsize=(6,4))      
plt.plot(np.linspace(10,40,2),R*(np.linspace(10,40,2)-np.mean(no_nan.bmi)),alpha=0.5)
sns.kdeplot(no_nan.bmi, no_nan.gpsbmi,n_levels=10,cmap='Blues',ax=ax,shade=True,shade_lowest=False)
plt.tight_layout()
fig=plt.gcf()
fig.savefig(f'/Users/nbaya/Documents/lab/risk_gradients/plots/bmi_bivariatekde.png',dpi=600)


# Plotting histograms
for disease, fullname in diseases.items():
    fig,ax = plt.subplots(figsize=(6,4))
    if disease=='bmi':
        plt.hist(no_nan['gps'+disease],50)
    else:
        plt.hist(df['gps'+disease],50)
    plt.title(f'Distribution of PGS for {fullname}')
    plt.xlabel('PGS')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.xlim([-5,5])
    plt.ylim([0,24000])
    fig=plt.gcf()
    fig.savefig(f'/Users/nbaya/Documents/lab/risk_gradients/plots/{disease}_hist.png',dpi=600)
    
# Make p-p plot
for disease, fullname in diseases.items():
    





