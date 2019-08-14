#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:01:22 2019

flip alleles for PRS-CS

@author: nbaya
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser(add_help=False)
#parser.add_argument('--path_to_ref', type=str, required=True, help="path to 1kg ref snp list")
#parser.add_argument('--path_to_ss', type=str, required=True, help="path to sumstats file")
#parser.add_argument('--flip_to_ref', type=int, required=True, help="whether to flip alleles to match 1kg ref or back to original")
parser.add_argument('--n_train_sub', type=int, required=True, help="number of individuals in sample")


args = parser.parse_args()

#path_to_ref = args.path_to_ref
#path_to_ss = args.path_to_ss
#flip_to_ref = bool(args.flip_to_ref)

n_train_sub = args.n_train_sub

wd = '/home/nbaya/'



#if flip_to_ref:
ref = pd.read_csv(wd+'ldblk_1kg_eur/snpinfo_1kg_hm3', delim_whitespace=True)
ss = pd.read_csv(wd+'data/ss.n_train_sub_{n_train_sub}.tsv', sep='\t')
bim = pd.read_csv(wd+'data/1kg_eur_hm3.bim',delim_whitespace=True,header=None,names=['CHR','SNP','CM','BP','A1','A2'])

merged = ss.merge(ref, on='SNP')

merged.loc[(merged.A1_x==merged.A2_y)&(merged.A2_x==merged.A1_y),'BETA'] = -merged.loc[(merged.A1_x==merged.A2_y)&(merged.A2_x==merged.A1_y),'BETA']

merged = merged.rename(columns={'A1_y':'A1','A2_y':'A2'})
merged[['SNP','A1','A2','BETA','P']].to_csv(wd+'data/ss.n_train_sub_{n_train_sub}.flipped.tsv',index=False,sep='\t')

merged = bim.merge(ref, on='SNP')

merged = merged.rename(columns={'A1_y':'A1','A2_y':'A2'})
merged[['CHR','SNP','CM','BP','A1','A2']].to_csv('/home/nbaya/data/1kg_eur_hm3.flipped.bim',index=False,header=None,sep='\t')





#else:
#    ref = pd.read_csv(path_to_ss, sep='\t', compression='gzip')
#    ss = pd.read_csv(path_to_ss, delim_whitespace=True)
#    merged[['SNP','A1','A2','BETA','P']].to_csv('ss.only_ref_snps.original.tsv',index=False,sep='\t')
    
    
    








# flip bim file
#ref = pd.read_csv('/home/nbaya/ldblk_1kg_eur/snpinfo_1kg_hm3',delim_whitespace=True)
#bim = pd.read_csv('/home/nbaya/data/1kg_eur_hm3.bim',delim_whitespace=True,header=None,names=['chr','SNP','cm','bp','A1','A2'])
#merged = bim.merge(ref, on='SNP')
#merged = merged.rename(columns={'A1_y':'A1','A2_y':'A2'})
#merged[['chr','SNP','cm','bp','A1','A2']].to_csv('/home/nbaya/data/1kg_eur_hm3.flipped.bim',index=False,header=None,sep='\t')


