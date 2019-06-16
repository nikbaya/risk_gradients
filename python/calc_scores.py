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
import os
from hail.utils.java import Env



parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phen_ls', nargs='+', type=str, required=True, help="phenotype code (e.g. for height, phen = 50")
#parser.add_argument('--n_ls', nargs='+', type=int, required=True, help="number of samples for each phenotype, ordered the same as phen_ls")
#parser.add_argument('--n_cas_ls', nargs='+', type=int, required=False, default=None, help="number of cases for each phenotype, ordered the same as phen_ls")
parser.add_argument('--frac_all_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of all individuals")
parser.add_argument('--frac_cas_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of cases")
parser.add_argument('--frac_con_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of controls")
parser.add_argument('--thresholds', nargs='+', type=float, required=False, default=None, help="thresholds for thresholding")
parser.add_argument('--seed', type=int, required=False, default=None, help="random seed for replicability")
args = parser.parse_args()

phen_ls = args.phen_ls
#n_ls = args.n_ls
#n_cas_ls = args.n_cas_ls
#n_cas_ls = n_ls if n_cas_ls == None else n_cas_ls
frac_all_ls = args.frac_all_ls
frac_cas_ls = args.frac_cas_ls
frac_con_ls = args.frac_con_ls
thresholds = args.thresholds
seed = args.seed
seed = 1 if seed is None else seed

frac_all_ls = [1] if frac_all_ls == None else frac_all_ls
frac_cas_ls = [1] if frac_cas_ls == None else frac_cas_ls
frac_con_ls = [1] if frac_con_ls == None else frac_con_ls
thresholds = [1] if thresholds == None else thresholds


wd = 'gs://nbaya/risk_gradients/'
local_wd = '/home/nbaya/' #working directory for VM instance
gwas_wd = wd+'gwas/'
phen_dict = {
    '50':'height',
    '2443':'diabetes',
    '21001':'bmi',
}

def get_mt(phen, variant_set, seed=None, test_set=0.1):
    start = dt.datetime.now()
    mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')

    print(f'\nReading UKB phenotype {phen_dict[phen]} (code: {phen})...')
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

    

    seed = seed if seed is not None else int(str(Env.next_seed())[:8])
    n_train = int(round(n*(1-test_set)))
    n_test = n-n_train
    print('\n###############')
    print(f'Setting {test_set} of total population to be in the testing set')
    print(f'n_train = {n_train}\tn_test = {n_test}')
    print(f'seed = {seed}')
    print('###############')
    randstate = np.random.RandomState(int(seed)) #seed random state for replicability
    labels = ['train']*n_train+['test']*n_test
    randstate.shuffle(labels)
    mt2 = mt1.add_col_index('tmp_index').key_cols_by('tmp_index')
    mt3 = mt2.annotate_cols(set = hl.literal(labels)[hl.int32(mt2.tmp_index)])
    mt3 = mt3.key_cols_by('s').drop('tmp_index')
    train_mt = mt3.filter_cols(mt3.set == 'train')
    test_mt = mt3.filter_cols(mt3.set == 'test')
    
    # Pruning test_mt to variants in variants table
    pruned_snps_file = 'ukb_imp_v3_pruned.bim'
    variants = hl.import_table(wd+pruned_snps_file,delimiter='\t',no_header=True,impute=True)
    print('\n###############')
    print(f'Pruning testing set to {variants.count()} variants in variants table...')
    print(f'Variants table used: {pruned_snps_file}')
    print('###############')
    variants = variants.rename({'f0':'chr','f1':'rsid','f3':'pos'}).key_by('rsid')
    test_mt = test_mt.key_rows_by('rsid')
    test_mt = test_mt.filter_rows(hl.is_defined(variants[test_mt.rsid])) #filter to variants defined in variants table
    
    
    n_cas_train = train_mt.filter_cols(train_mt.phen == 1).count_cols()
    n_cas_test = test_mt.filter_cols(test_mt.phen == 1).count_cols()

    elapsed = dt.datetime.now() - start
    print('\n###############')
    print(f'Original prevalence of {phen_dict[phen]} (code: {phen}): {round(n_cas/n,6)}')
    print(f'Prevalence in training dataset: {round(n_cas_train/n_train,6)}')
    print(f'Prevalence in testing dataset: {round(n_cas_test/n_test,6)}')
    print(f'(Note: If trait is not binary, these will probably all be 0)')
    print(f'Time to get training and testing sets: {round(elapsed.seconds/60, 2)} minutes')
    print('###############')

    return train_mt, n_train, n_cas_train, test_mt, n_test, n_cas_test



if __name__=="__main__":
    header =  '\n############\n'
    header += f'Phenotypes: {[phen_dict[phen]+f" (code: {phen})" for phen in phen_ls]}\n'
    header += f'Downsampling fractions for all: {frac_all_ls}\n' if frac_all_ls != None else ''
    header += f'Downsampling fractions for cases: {frac_cas_ls}\n' if frac_cas_ls != None else ''
    header += f'Downsampling fractions for controls: {frac_con_ls}\n' if frac_con_ls != None else ''
    header += f'Thresholds: {thresholds}\n'
    header += f'Random seed: {seed}\n'
    header += '############'
    print(header)
    
    for i, phen in enumerate(phen_ls):
        _, n, n_cas, test_mt, n_test, n_cas_test= get_mt(phen, 'hm3', seed=seed) 
        for frac_all in frac_all_ls:
            n_new0 = int(frac_all*n)
            for frac_cas in frac_cas_ls:
                n_cas_new = int(frac_cas*n_cas)
                n_new1 = (n_new0 - n_cas) + n_cas_new
                for frac_con in frac_con_ls:
                    n_new = int(frac_con*(n_new1-n_cas_new)+n_cas_new)
                    if frac_con == 1 and frac_cas ==1:
                        suffix = f'.{phen}.n_{n_new}of{n}.seed_{seed}'
                    else:
                        suffix = f'.{phen}.n_{n_new}of{n}.n_cas_{n_cas_new}of{n_cas}.seed_{seed}'
                    
                    print('Importing GWAS results table...')
                    gwas = hl.import_table(f'{gwas_wd}ss{suffix}.tsv.bgz',force_bgz=True,impute=True)
                    gwas = gwas.key_by('SNP')
                    
                    print('Annotating test set matrix table with betas, p-values...')
                    test_mt1 = test_mt.annotate_rows(beta = gwas[test_mt.rsid].eff,
                                                     pval = gwas[test_mt.rsid].pval)
                    
                    for t in thresholds:
                        # Thresholding to variants with p-value less than threshold
                        print('\n###############')
                        print(f'Filtering variants by p-value threshold = {t}...')
                        print('###############')
                        if t != 1:
                            test_mt1 = test_mt1.filter_rows(test_mt1.pval < t)
                        n_rows = test_mt1.count_rows()
                        print('\n###############')
                        print(f'Number of variants remaining after thresholding: {n_rows}...')
                        print('###############')

                        print('Calculating PGS...')
                        print(f'frac_all: {frac_all}\tfrac_cas: {frac_cas}\tfrac_con: {frac_con}')
                        print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(dt.datetime.now()))
                        start_pgs = dt.datetime.now()
                        test_mt2 = test_mt1.annotate_cols(pgs = hl.agg.sum(test_mt1.dosage*test_mt1.beta))
                        test_mt2.cols().key_by('s').select('phen','pgs').export(f'{wd}pgs{suffix}.threshold_{t}.tsv.bgz')
                        elapsed_pgs = dt.datetime.now()-start_pgs
                        print('\n###############')
                        print('\nFinished calculating PGS')
                        print(f'Time for calculating PGS: {round(elapsed_pgs.seconds/60, 2)} minutes')
                        print('###############')
        
                        print('Calculating R^2 between PGS and phenotype...')
                        print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(dt.datetime.now()))
                        if not os.path.isdir(local_wd):
                            os.mkdir(local_wd)
                        subprocess.call(['gsutil','cp',f'{wd}pgs{suffix}.tsv.bgz',local_wd])
                        df = pd.read_csv(f'{local_wd}pgs{suffix}.tsv.bgz',delimiter='\t',compression='gzip')
                        r, pval = stats.pearsonr(df.pgs, df.phen)
                        print('\n###########################')
                        print(f'PGS-phenotype correlation for all {n_test} individuals in test set')
                        print(f'r = {r}, pval = {pval}')
                        print('###########################')
                        result = pd.DataFrame(data={'phen': [phen],'desc':phen_dict[phen],
                                                    'n_train':[n],'n_cas_train':[n_cas],
                                                    'n_test':[n_test],'n_cas_test':[n_cas_test],
                                                    'r':[r],'pval':[pval]})
                        corr_file = f'{local_wd}corr{suffix}.threshold_{t}.tsv'
                        result.to_csv(path_or_buf=corr_file,sep='\t',index=False)
                        subprocess.call(['gsutil','cp',corr_file,wd])
                        print('\nFinished calculating R^2')

