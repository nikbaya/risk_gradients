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

cloud_wd = 'gs://nbaya/risk_gradients/'
gwas_wd = cloud_wd+'gwas/'
phen_dict = {
    '50':'height',
    '2443':'diabetes',
    '21001':'bmi',
}

local_wd = '/Users/nbaya/Documents/lab/risk_gradients'


phen = '50'
n = 360338
n_cas = 0
frac_all_ls = [0.2, 0.4, 0.6, 0.8, 1]
frac_cas_ls = [1]
frac_con_ls = [1]
seed = 1


#df = pd.DataFrame(columns=['phen','desc','n','n_new','n_cas','n_cas_new','r_all','pval_all','r_sub','pval_sub'])


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
            subprocess.call(['gsutil','cp',f'{cloud_wd}pgs{suffix}.tsv.bgz','/home/nbaya/'])
            subprocess.call(['gsutil','cp',f'{gwas_wd}iid{suffix}.tsv.bgz','/home/nbaya/'])
            df = pd.read_csv(f'/home/nbaya/pgs{suffix}.tsv.bgz',delimiter='\t',compression='gzip')
            r_all, pval_all = stats.pearsonr(df.pgs, df.phen)
            iid = pd.read_csv(f'/home/nbaya/iid{suffix}.tsv.bgz',delimiter='\t',compression='gzip')
            df1 = df[df.s.isin(iid.iid.tolist())]
            r_sub, pval_sub = stats.pearsonr(df1.pgs,df1.phen)
            array = [[r_all, pval_all],[r_sub, pval_sub]]
            np.savetxt(f'/home/nbaya/corr{suffix}.txt',array,delimiter='\t')
            subprocess.call(['gsutil','cp',f'/home/nbaya/corr{suffix}.txt',cloud_wd])
#            corr0 = subprocess.Popen(['gsutil','cat',f'{cloud_wd}corr{suffix}.txt'],stdout=subprocess.PIPE).communicate()[0]
#            corr1 = corr0.decode('utf-8')
#            corr2 = corr1.replace('\t','\n')
#            corr = corr2.split('\n')
#            r_all, pval_all, r_sub, pval_sub = float(corr[0]), float(corr[1]), float(corr[2]), float(corr[3])
#            df.loc[0] = [phen, phen_dict[phen], n, n_new, n_cas, n_cas_new, r_all, pval_all, r_sub, pval_sub]
#            i += 1
            
#plt.plot(df.n_new, df.r_all**2,'.')
#plt.plot(df.n_new, df.r_sub**2,'.')
#plt.legend(['r^2 for all individuals','r^2 for subset of individuals'])