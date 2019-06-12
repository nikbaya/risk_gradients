#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:39:46 2019

Calculate PGS from summary statistics and genotypes, pruning to certain variants 

@author: nbaya
"""

import hail as hl
import argparse
import scipy.stats as stats
import numpy as np
import subprocess
import pandas as pd
import datetime as dt
from hail.utils.java import Env


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phen_ls', nargs='+', type=str, required=True, help="phenotype code (e.g. for height, phen = 50")
#parser.add_argument('--n_ls', nargs='+', type=int, required=True, help="number of samples for each phenotype, ordered the same as phen_ls")
#parser.add_argument('--n_cas_ls', nargs='+', type=int, required=False, default=None, help="number of cases for each phenotype, ordered the same as phen_ls")
parser.add_argument('--frac_all_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of all individuals")
parser.add_argument('--frac_cas_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of cases")
parser.add_argument('--frac_con_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of controls")
parser.add_argument('--seed', type=int, required=False, default=None, help="random seed for replicability")
args = parser.parse_args()

phen_ls = args.phen_ls
#n_ls = args.n_ls
#n_cas_ls = args.n_cas_ls
#n_cas_ls = n_ls if n_cas_ls == None else n_cas_ls
frac_all_ls = args.frac_all_ls
frac_cas_ls = args.frac_cas_ls
frac_con_ls = args.frac_con_ls
seed = args.seed
seed = 1 if seed is None else seed

frac_all_ls = [1] if frac_all_ls == None else frac_all_ls
frac_cas_ls = [1] if frac_cas_ls == None else frac_cas_ls
frac_con_ls = [1] if frac_con_ls == None else frac_con_ls


wd = 'gs://nbaya/risk_gradients/'
gwas_wd = wd+'gwas/'
phen_dict = {
    '50':'height',
    '2443':'diabetes',
    '21001':'bmi',
}

def get_mt(phen, variant_set):
    mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')
    
    print(f'\nReading UKB phenotype {phen_dict[phen]} (code: {phen})...')
    #        mt0 = hl.read_matrix_table('gs://nbaya/split/ukb31063.hm3_variants.gwas_samples_v2.mt') #old version
        
    phen_tb0 = hl.import_table('gs://phenotype_31063/ukb31063.phesant_phenotypes.both_sexes.tsv.bgz',
                               missing='',impute=True,types={'"userId"': hl.tstr}).rename({ '"userId"': 's', '"'+phen+'"': 'phen'})
    phen_tb0 = phen_tb0.key_by('s')
    phen_tb = phen_tb0.select(phen_tb0['phen'])    
    
    mt1 = mt0.annotate_cols(phen_str = hl.str(phen_tb[mt0.s]['phen']).replace('\"',''))
    mt1 = mt1.filter_cols(mt1.phen_str == '',keep=False)
    
    if phen_tb.phen.dtype == hl.dtype('bool'):
        mt1 = mt1.annotate_cols(phen = hl.bool(mt1.phen_str)).drop('phen_str')
    else:
        mt1 = mt1.annotate_cols(phen = hl.float64(mt1.phen_str)).drop('phen_str')            
    
    #Remove withdrawn samples
    withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
    withdrawn_set = set(withdrawn.f0.take(withdrawn.count()))
    mt1 = mt1.filter_cols(hl.literal(withdrawn_set).contains(mt1['s']),keep=False) 
    mt1 = mt1.key_cols_by('s')
    
    n = mt1.count_cols()
    n_cas = mt1.filter_cols(mt1.phen == 1).count_cols()
    
    variants = hl.import_table(wd+'ukb_imp_v3_pruned.bim',delimiter='\t',no_header=True,impute=True)
    variants = variants.rename({'f0':'chr','f1':'rsid','f3':'pos'}).key_by('rsid')
    
    mt2 = mt1.key_rows_by('rsid')
    mt2 = mt2.filter_rows(hl.is_defined(variants[mt2.rsid])) #filter to variants in variants table

    return mt2, n, n_cas

def downsample(mt, frac, phen, for_cases=None, seed = None):
    start = dt.datetime.now()
    assert frac <= 1 and frac >= 0, "frac must be in [0,1]"
    phen_name = phen._ir.name
    n = mt.count_cols()
    n_cas = mt.filter_cols(mt[phen_name]==1).count_cols()
    if frac == 1:
        return mt, n, n_cas
    else:
        seed = seed if seed is not None else int(str(Env.next_seed())[:8])
        header = '\n************\n'
        header += 'Downsampling '+('all' if for_cases is None else ('cases'*for_cases+'controls'*(for_cases==0)))+f' by frac = {frac}\n'
        header += f'n: {n}\n'
        header += f'n_cas: {n_cas}\nn_con: {n-n_cas}\nprevalence: {round(n_cas/n,6)}\n' if for_cases != None else ''
        header += f'seed: {seed}\n'
        header += '************'
        print(header)
        col_key = mt.col_key
        randstate = np.random.RandomState(int(seed)) #seed random state for replicability
        for_cases = bool(for_cases) if for_cases != None else None
        filter_arg = (mt[phen_name] == (for_cases==0)) if for_cases != None else (hl.is_defined(mt[phen_name])==False)
        mtA = mt.filter_cols(filter_arg) #keep all individuals in this mt
        mtB = mt.filter_cols(filter_arg , keep=False) #downsample individuals in this mt
        mtB = mtB.add_col_index('col_idx_tmpB')
        mtB = mtB.key_cols_by('col_idx_tmpB')
        nB = n_cas*for_cases + (n-n_cas)*(for_cases==0) if for_cases is not None else n
        n_keep = int(nB*frac)
        labels = ['A']*(n_keep)+['B']*(nB-n_keep)
        randstate.shuffle(labels)
        mtB = mtB.annotate_cols(label = hl.literal(labels)[hl.int32(mtB.col_idx_tmpB)])
        mtB = mtB.filter_cols(mtB.label == 'A')
        mtB = mtB.key_cols_by(*col_key)
        mtB = mtB.drop('col_idx_tmpB','label')
        mt1 = mtA.union_cols(mtB) 
        n_new = mt1.count_cols()
        n_cas_new = mt1.filter_cols(mt1[phen_name]==1).count_cols()
        elapsed = dt.datetime.now()-start
        print('\n************')
        print('Finished downsampling '+('all' if for_cases is None else ('cases'*for_cases+'controls'*(for_cases==0)))+f' by frac = {frac}')
        print(f'n: {n} -> {n_new} ({round(100*n_new/n,3)}% of original)')
        if n_cas != 0 and n_new != 0 :
            print(f'n_cas: {n_cas} -> {n_cas_new} ({round(100*n_cas_new/n_cas,3)}% of original)')
            print(f'n_con: {n-n_cas} -> {n_new-n_cas_new} ({round(100*(n_new-n_cas_new)/(n-n_cas),3)}% of original)')
            print(f'prevalence: {round(n_cas/n,6)} -> {round(n_cas_new/n_new,6)} ({round(100*(n_cas_new/n_new)/(n_cas/n),6)}% of original)')
        print(f'Time for downsampling: '+str(round(elapsed.seconds/60, 2))+' minutes')
        print('************')
        return mt1, n_new, n_cas

if __name__=="__main__":
    
    for i, phen in enumerate(phen_ls):
        mt1, n, n_cas = get_mt(phen, 'hm3')
        for frac_all in frac_all_ls:
            n_new0 = int(frac_all*n)
            for frac_cas in frac_cas_ls:
                n_cas_new = int(frac_cas*n_cas)
                for frac_con in frac_con_ls:
                    n_new = int(frac_con*(n_new0-n_cas_new)+n_cas_new)
                    if frac_con == 1 and frac_cas ==1:
                        suffix = f'.{phen}.n_{n_new}of{n}.seed_{seed}'
                    else:
                        suffix = f'.{phen}.n_{n_new}of{n}.n_cas_{n_cas_new}of{n_cas}.seed_{seed}'
                    
                    print('importing table...')
                    gwas = hl.import_table(f'{gwas_wd}ss{suffix}.tsv.bgz',force_bgz=True,impute=True)
                    gwas = gwas.key_by('SNP')
                    
                    print('annotating with betas...')
                    mt2 = mt1.annotate_rows(beta = gwas[mt1.rsid].eff)
                    
                    print('calculating PGS...')
                    print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(dt.datetime.now()))
                    start_pgs = dt.datetime.now()
                    mt3 = mt2.annotate_cols(pgs = hl.agg.sum(mt2.dosage*mt2.beta))
                    mt3.cols().select('phen','pgs').export(f'{wd}pgs{suffix}.tsv.bgz')
                    elapsed_pgs = dt.datetime.now()-start_pgs
                    print('\nFinished calculating PGS')
                    print(f'Time for calculating PGS: {round(elapsed_pgs.seconds/60, 2)}minutes')
    
                    print('calculating R^2 between PGS and phenotype...')
                    print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(dt.datetime.now()))
                    start_r2 = dt.datetime.now()
                    subprocess.call(['gsutil','cp',f'{wd}pgs{suffix}.tsv.bgz','/home/nbaya/'])
                    subprocess.call(['gsutil','cp',f'{gwas_wd}iid{suffix}.tsv.bgz','/home/nbaya/'])
                    df = pd.read_csv(f'/home/nbaya/pgs{suffix}.tsv.bgz',delimiter='\t',compression='gzip')
                    r_all, pval_all = stats.pearsonr(df.pgs, df.phen)
                    iid = pd.read_csv(f'/home/nbaya/iid{suffix}.tsv.bgz',delimiter='\t',compression='gzip')
                    df1 = df[df.s.isin(iid.iid.tolist())]
                    r_sub, pval_sub = stats.pearsonr(df1.pgs,df1.phen)
                    print('\n****************************')
                    print(f'PGS x phenotype correlation for all {n} individuals')
                    print(f'r = {r_all}, pval = {pval_all}')
                    print(f'PGS x phenotype correlation for all {n_new} individuals')
                    print(f'r = {r_sub}, pval = {pval_sub}')
                    print('****************************')
                    array = [[r_all, pval_all],[r_sub, pval_sub]]
                    np.savetxt(f'/home/nbaya/corr{suffix}.txt',array,delimiter='\t')
                    subprocess.call(['gsutil','cp',f'/home/nbaya/corr{suffix}.txt',wd])
                    elapsed_r2 = dt.datetime.now()-start_r2
                    print('\nFinished calculating R^2')
                    print(f'Time for calculating R^2: {round(elapsed_r2.seconds/60, 2)}minutes')