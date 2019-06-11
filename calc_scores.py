#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:39:46 2019

Calculate PGS from summary statistics and genotypes, pruning to certain variants 

@author: nbaya
"""

import hail as hl
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phen_ls', nargs='+', type=str, required=True, help="phenotype code (e.g. for height, phen = 50")
parser.add_argument('--n_ls', nargs='+', type=int, required=True, help="number of samples for each phenotype, ordered the same as phen_ls")
parser.add_argument('--n_cas_ls', nargs='+', type=int, required=False, default=None, help="number of cases for each phenotype, ordered the same as phen_ls")
parser.add_argument('--frac_all_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of all individuals")
parser.add_argument('--frac_cas_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of cases")
parser.add_argument('--frac_con_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of controls")
parser.add_argument('--seed', type=int, required=False, default=None, help="random seed for replicability")
args = parser.parse_args()

phen_ls = args.phen_ls
n_ls = args.n_ls
n_cas_ls = args.n_cas_ls
n_cas_ls = n_ls if n_cas_ls == None else n_cas_ls
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

mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.hm3_variants.gwas_samples_repart.mt')

variants = hl.import_table(wd+'ukb_imp_v3_pruned.bim',delimiter='\t',no_header=True,impute=True)
variants = variants.rename({'f0':'chr','f1':'rsid','f3':'pos'}).key_by('rsid')

mt1 = mt0.key_rows_by('rsid')
mt1 = mt1.filter_rows(hl.is_defined(variants[mt1.rsid])) #filter to variants in variants table

for i, phen in enumerate(phen_ls):
    n = int(n_ls[i])
    n_cas = int(n_cas_ls[i])
    for frac_all in frac_all_ls:
        for frac_cas in frac_cas_ls:
            for frac_con in frac_con_ls:
                n_new = int(frac_all*n)
                n_cas_new = int(frac_cas*n_cas)
                if frac_con == 1 and frac_cas ==1:
                    gwas_path = f'{gwas_wd}ss.{phen}.n_{n_new}of{n}.seed_{seed}.tsv.bgz' 
                else:
                    gwas_path = f'{gwas_wd}ss.{phen}.n_{n_new}of{n}.n_cas_{n_cas_new}of{n_cas}.seed_{seed}.tsv.bgz' 
                print('importing table...')
                gwas = hl.import_table(gwas_path,force_bgz=True,impute=True)
                gwas = gwas.key_by('SNP')
                
                print('annotating with betas...')
                mt2 = mt1.annotate_rows(beta = gwas[mt1.rsid].eff)
                print('calculating dot product of genotypes with betas...')
                mt3 = mt2.annotate_cols(pgs = hl.agg.sum(mt2.dosage*mt2.beta))
                if frac_con == 1 and frac_cas ==1:
                    pgs_path = f'{wd}pgs.{phen}.n_{n_new}of{n}.seed_{seed}.tsv.bgz' 
                else:
                    pgs_path = f'{wd}pgs.{phen}.n_{n_new}of{n}.n_cas_{n_cas_new}of{n_cas}.seed_{seed}.tsv.bgz' 
                mt3.cols().select('pgs').export(pgs_path)
                