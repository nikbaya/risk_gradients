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
import datetime as dt
import subprocess


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phen_ls', nargs='+', type=str, required=True, help="phenotype code (e.g. for height, phen = 50")
parser.add_argument('--frac_all_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of all individuals")
parser.add_argument('--frac_cas_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of cases")
parser.add_argument('--frac_con_ls', nargs='+', type=float, required=False, default=None, help="downsampling fraction of controls")
parser.add_argument('--seed', type=int, required=False, default=1, help="random seed for replicability")
args = parser.parse_args()

phen_ls = args.phen_ls
frac_all_ls = args.frac_all_ls
frac_cas_ls = args.frac_cas_ls
frac_con_ls = args.frac_con_ls
seed = args.seed

frac_all_ls = [1] if frac_all_ls == None else frac_all_ls
frac_cas_ls = [1] if frac_cas_ls == None else frac_cas_ls
frac_con_ls = [1] if frac_con_ls == None else frac_con_ls

variant_set = 'hm3'
phen_dict = {
    '50':'height',
    '2443':'diabetes',
    '21001':'bmi',
}
gwas_wd = 'gs://nbaya/risk_gradients/gwas/'

def get_mt(phen, variant_set, seed=None, test_set=0.1):
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

    #Create training and testing matrix tables
    seed = seed if seed is not None else int(str(Env.next_seed())[:8])
    n_train = int(round(n*(1-test_set)))
    n_test = n-n_train
    print('*****************')
    print(f'Setting {test_set} of total population to be in the testing set')
    print(f'n_train = {n_train}\tn_test = {n_test}')
    print(f'seed = {seed}')
    print('*****************')
    randstate = np.random.RandomState(int(seed)) #seed random state for replicability
    labels = ['train']*n_train+['test']*n_test
    randstate.shuffle(labels)
    mt2 = mt1.add_col_index('tmp_index').key_cols_by('tmp_index')
    mt3 = mt2.annotate_cols(set = hl.literal(labels)[hl.int32(mt2.tmp_index)])
    # if not train_mt_done:
    # mt3.filter_cols(mt3.label == 'train').write(train_mt_path,overwrite=True)
    # if not test_mt_done:
    # mt3.filter_cols(mt3.label == 'test').write(test_mt_path, overwrite=True)
    # train_mt = hl.read_matrix_table(train_mt_path)
    # test_mt = hl.read_matrix_table(test_mt_path)
    train_mt = mt3.filter_cols(mt3.set == 'train')
    test_mt = mt3.filter_cols(mt3.set == 'test')

    n_cas_train = train_mt.filter_cols(train_mt.phen == 1).count_cols()
    n_cas_test = test_mt.filter_cols(test_mt.phen == 1).count_cols()

    print('*****************')
    print(f'Original prevalence of {phen_dict[phen]} (code: {phen}): {round(n_cas/n,6)}')
    print(f'Prevalence in training dataset: {round(n_cas_train/n_train,6)}')
    print(f'Prevalence in testing dataset: {round(n_cas_test/n_test,6)}')
    print(f'If trait is not case/control, these will probably all be 0.')
    print('*****************')

    return train_mt, n_train, n_cas_train, test_mt, n_test, n_cas_test

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


if __name__ == "__main__":
    header =  '\n*************\n'
    header += f'Phenotypes to downsample: {[phen_dict[phen]+f" (code: {phen})" for phen in phen_ls]}\n'
    header += f'Downsampling fractions for all: {frac_all_ls}\n' if frac_all_ls != None else ''
    header += f'Downsampling fractions for cases: {frac_cas_ls}\n' if frac_cas_ls != None else ''
    header += f'Downsampling fractions for controls: {frac_con_ls}\n' if frac_con_ls != None else ''
    header += f'Random seed: {seed}\n'
    header += '*************'
    print(header)

    for phen in phen_ls:
        print('\n*************')
        print(f'Starting phenotype: {phen_dict[phen]} (code: {phen})')
        print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(dt.datetime.now()))
        print('*************')
        mt, n, n_cas, _, _, _ = get_mt(phen, variant_set, seed=seed, )

        start_phen = dt.datetime.now()

        for frac_all in frac_all_ls:
            mt1, _, _ = downsample(mt=mt,frac=frac_all,phen=mt.phen,for_cases=None,seed=seed)
            for frac_cas in frac_cas_ls:
                mt2, _, _ = downsample(mt=mt1,frac=frac_cas,phen=mt1.phen,for_cases=1,seed=seed)
                # if len(frac_cas_ls)>1 or frac_cas_ls != [1]:
                #     print('\ncheckpointing matrix table')
                #     mt2 = mt2.checkpoint(f'{gwas_wd}tmp.mt')
                for frac_con in frac_con_ls:
                    start_iter = dt.datetime.now()
                    mt3, n_new, n_cas_new = downsample(mt=mt2,frac=frac_con,phen=mt2.phen,for_cases=0,seed=seed)

                    gt0 = hl.read_matrix_table('gs://phenotype_31063/hail/imputed/ukb31063.GT.autosomes.mt/')
                    mt3 = mt3.annotate_entries(GT = gt0[(mt3.locus,mt3.alleles),mt3.s].GT)
                    mt3 = mt3.annotate_rows(maf = hl.agg.stats(mt3.dosage).mean/2) # annotate with maf
                    if frac_con == 1 and frac_cas ==1:
                        id_path = f'{gwas_wd}iid.{phen}.n_{n_new}of{n}.seed_{seed}.tsv.bgz'
                    else:
                        id_path = f'{gwas_wd}iid.{phen}.n_{n_new}of{n}.n_cas_{n_cas_new}of{n_cas}.seed_{seed}.tsv.bgz'
                    cols = mt3.cols().key_by().select('s')
                    cols = cols.annotate(iid=cols.s).annotate(s=1).rename({'s':'fid'}).export(id_path)

                    cov_list = [ mt3['isFemale'], mt3['age'], mt3['age_squared'], mt3['age_isFemale'],
                        mt3['age_squared_isFemale'] ]+ [mt3['PC{:}'.format(i)] for i in range(1, 21)]

                    ht = hl.linear_regression_rows(
                            y=mt3.phen,
                            x=mt3.dosage,
                            covariates=[1]+cov_list,
                            pass_through = ['rsid','maf'])

                    ht = ht.rename({'rsid':'SNP'}).key_by('SNP')

                    ss_template = hl.read_table('gs://nbaya/rg_sex/hm3.sumstats_template.ht/')
                    ss_template  = ss_template.key_by('SNP')

                    ss = ss_template.annotate(N = n_new)
                    locus = hl.str(ht[ss.SNP].locus).split(':')
                    ss = ss.annotate(ch = locus[0],
                                     pos = locus[1].split('_')[0],
                                     pval = ht[ss.SNP]['p_value'],
                                     maf = ht[ss.SNP]['maf'],
                                     eff = ht[ss.SNP]['beta'])

                    if frac_con == 1 and frac_cas ==1:
                        path = f'{gwas_wd}ss.{phen}.n_{n_new}of{n}.seed_{seed}.tsv.bgz'
                    else:
                        path = f'{gwas_wd}ss.{phen}.n_{n_new}of{n}.n_cas_{n_cas_new}of{n_cas}.seed_{seed}.tsv.bgz'
                    ss.export(path)
                    elapsed_iter = dt.datetime.now()-start_iter
                    print('\n*************')
                    print(f'Finished GWAS for downsampled phenotype: {phen_dict[phen]} (code: {phen})')
                    print(f'frac_all = {frac_all}, frac_cas = {frac_cas}, frac_con = {frac_con}')
                    print(f'Time for downsampled GWAS: '+str(round(elapsed_iter.seconds/60, 2))+' minutes')
                    print('*************')

        elapsed_phen = dt.datetime.now()-start_phen
        print('\n*************')
        print(f'Finished phenotype: {phen_dict[phen]} (code: {phen})')
        print(f'Number of downsampling fraction combinations: {len(frac_all_ls)*len(frac_con_ls)*len(frac_cas_ls)}')
        print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(dt.datetime.now()))
        print(f'Time for phenotype: '+str(round(elapsed_phen.seconds/60, 2))+' minutes')
        print('*************')