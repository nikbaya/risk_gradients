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

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phen', type=str, required=True, help="phenotype code (e.g. for height, phen = 50")
parser.add_argument('--variant_set', type=str, required=True, help="set of variants to use")

phen = args.phen
    desc = phen_dict[phen]
    batch = args.batch
    n_chunks = args.n_chunks #number of subgroups (or "chunks")
    variant_set = str(args.variant_set)

variant_set = 'hm3'
phen_dict = {
    '50':'height',
    '2443':'diabetes',
    '21001':'bmi',
}

mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')

print('\nReading UKB phenotype...')
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

