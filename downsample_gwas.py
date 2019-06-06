#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:41:38 2019

Run downsampled GWAS on selected phenotypes

Also options for varying case/control ratios

@author: nbaya
"""

import hail as hl
import argparse
import numpy as np
from hail.utils.java import Env


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phen_ls', nargs='+', type=str, required=True, help="phenotype code (e.g. for height, phen = 50")
parser.add_argument('--frac_all_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of all individuals")
parser.add_argument('--frac_cas_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of cases")
parser.add_argument('--frac_con_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of controls")
args = parser.parse_args()

phen_ls = args.phen_ls
frac_all_ls = args.frac_all_ls
frac_cas_ls = args.frac_cas_ls
frac_con_ls = args.frac_con_ls

variant_set = 'hm3'
phen_dict = {
    '50':'height',
    '2443':'diabetes',
    '21001':'bmi',
}
gc_bucket = 'gs://nbaya/risk_gradients/gwas/'

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
    
    return mt1
    
def downsample(mt, frac, phen, for_cases=None, seed = None):
    assert frac <= 1 and frac >= 0, "frac must be in [0,1]"
    phen_name = phen._ir.name
    n = mt.count_cols()
    n_cas = mt.filter_cols(mt[phen_name]==1).count_cols()
    if frac == 1:
        return mt, n, n_cas
    else:
        print('\n************')
        print('Downsampling '+('all' if for_cases is None else ('cases'*for_cases+'controls'*(for_cases==0)))+f' by frac = {frac}')
        print(f'n: {n}')
        print(f'n_cas: {n_cas}\nn_con: {n-n_cas}\nprevalence: {round(n_cas/n,3)}' if for_cases != None else '')
        print('************')
        col_key = mt.col_key
        seed = seed if seed is not None else int(str(Env.next_seed())[:8])
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
        print('\n************')
        print(f'n: {n} -> {n_new} ({round(100*n_new/n,3)}%)')
        print(f'n_cas: {n_cas} -> {n_cas_new} ({round(100*n_cas_new/n_cas,3)}% of original)')
        print(f'n_con: {n-n_cas} -> {n_new-n_cas_new} ({round(100*(n_new-n_cas_new)/(n-n_cas),3)}% of original)')
        print(f'prevalence: {round(n_cas/n,3)} -> {round(n_cas_new/n_new,3)} ({round(100*(n_cas_new/n_new)/(n/n_cas),3)}% of original)')
        print('************')
        return mt1, n_new, n_cas


if __name__ == "__main__":    
    header =  '\n*************\n'
    header += f'Phenotypes to downsample: {[phen_dict[phen]+f" (code: {phen})" for phen in phen_ls]}'
    header += f'Downsampling fractions for all: {frac_all_ls}\n' if frac_all_ls != None else ''
    header += f'Downsampling fractions for cases: {frac_cas_ls}\n' if frac_cas_ls != None else ''
    header += f'Downsampling fractions for controls: {frac_con_ls}\n' if frac_con_ls != None else ''
    header += '*************'
    print(header)
    
    frac_all_ls = [1] if frac_all_ls == None else frac_all_ls
    frac_cas_ls = [1] if frac_cas_ls == None else frac_cas_ls
    frac_con_ls = [1] if frac_con_ls == None else frac_con_ls
    
    for phen in phen_ls:
        header =  '\n*************\n'
        print(f'Starting phenotype: {phen_dict[phen]} (code: {phen})') 
        header += '*************'
        mt = get_mt(phen, variant_set)
    
        cov_list = [ mt['isFemale'], mt['age'], mt['age_squared'], mt['age_isFemale'],
                        mt['age_squared_isFemale'] ]+ [mt['PC{:}'.format(i)] for i in range(1, 21)] 
        
        for frac_all in frac_all_ls: #for each downsampling fraction in frac_all_ls
            mt1, _, _ = downsample(mt=mt,frac=frac_all,phen=mt.phen,for_cases=None)
            for frac_cas in frac_cas_ls:
                mt2, _, _ = downsample(mt=mt1,frac=frac_cas,phen=mt1.phen,for_cases=1)
                for frac_con in frac_con_ls:
                    mt3, n, n_cas = downsample(mt=mt2,frac=frac_con,phen=mt2.phen,for_cases=0)
                    ht = hl.linear_regression_rows(
                            y=mt.phen,
                            x=mt.dosage,
                            covariates=[1]+cov_list,
                            pass_through = ['rsid'])
                
                    ht = ht.rename({'rsid':'SNP'}).key_by('SNP')
    
                    ss_template = hl.read_table('gs://nbaya/rg_sex/hm3.sumstats_template.ht/')
                    ss_template  = ss_template .key_by('SNP')
                    
                    ss = ss_template.annotate(N = n)
                    ss = ss.annotate(Z = ht[ss.SNP]['Z'])
                    
                    if frac_con == 1 and frac_cas ==1:
                        path = f'{gc_bucket}{phen}.downsampled.n_{n}.tsv.bgz' 
                    else:
                        path = f'{gc_bucket}{phen}.downsampled.n_{n}.n_cas_{n_cas}.tsv.bgz' 
                    ss.export(path)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        