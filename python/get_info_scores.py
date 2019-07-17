#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:07:43 2019

Collect info scores for use in LDpred.

Uses code written by Liam Abbott.

@author: nbaya
"""

import hail as hl
import pandas as pd

def get_info():
    contigs = [str(x) for x in range(1, 23)] + ['X', 'XY']
    hts = []
    
    for c in contigs:
        ht = hl.import_table('gs://fc-7d5088b4-7673-45b5-95c2-17ae00a04183/imputed/ukb_mfi_chr{}_v3.txt'.format(c), no_header=True)
        ht = ht.rename({'f0': 'varid',
                        'f1': 'rsid',
                        'f2': 'pos',
                        'f3': 'ref_allele',
                        'f4': 'alt_allele',
                        'f5': 'maf',
                        'f6': 'minor_allele',
                        'f7': 'info'})
        c = 'X' if c == 'XY' else c
        ht = ht.annotate(locus=hl.locus(hl.literal(c), hl.int(ht.pos)),
                         alleles=hl.array([ht.ref_allele, ht.alt_allele]),
                         maf=hl.float(ht.maf),
                         info=hl.float(ht.info))
        ht = ht.select('locus', 'alleles', 'varid', 'rsid', 'minor_allele', 'maf', 'info')
        ht = ht.key_by('locus', 'alleles')
        hts.append(ht)
    
    ht_union = hl.Table.union(*hts)
    ht_union.export('gs://nbaya/risk_gradients/ukb.maf_info.tsv.bgz')
    
    ht = hl.read_table('gs://nbaya/risk_gradients/ukb.maf_info.ht')
    ht.describe()
    ht.show()
    
    # wrote hm3 subset to 'gs://nbaya/risk_gradients/ukb.maf_info.hm3.ht'
    
def add_info():
    info = pd.read_csv('ukb.maf_info.tsv.bgz',compression='gzip',sep='\t')
    ss = pd.read_csv('ss.50.n_324304of324304.seed_1.tsv.bgz',compression='gzip',sep='\t')
    ss1 = ss.merge(info, left_on='SNP',right_on='rsid',suffixes=('','2'))