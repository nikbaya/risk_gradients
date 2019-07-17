#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:41:51 2019

@author: nbaya
"""

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
import os


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
    '50':['height', 360338, 0, 324304, 0],
    '2443':['diabetes', 360142, 17272, 324128, 15494],
    '21001':['BMI',359933, 0, 323940, 0]
}
wd = 'gs://nbaya/risk_gradients/' #working directory in cloud
local_wd = '/home/nbaya/' #working directory for VM instance
gwas_wd = wd+'gwas/'

def get_plink(phen, variant_set, test_set=0.1, get='both', overwrite=False, seed=None):
    print('\n###############')
    print(f'phen: {phen_dict[phen][0]} (code: {phen})')
    print(f'get: {"training and testing sets" if get=="both" else get+"ing set"}')
    print(f'overwrite: {overwrite}')
    print(f'seed: {seed}')
    print('###############')
    start = dt.datetime.now()
    if not os.path.isdir(local_wd):
        os.mkdir(local_wd)
    train_mt_path = wd+f'train1k.{phen}.seed_{seed}'
    test_mt_path = wd+f'test1k.{phen}.seed_{seed}'
    train_success = f'{local_wd}train.{phen}.success'
    test_success = f'{local_wd}test.{phen}.success'
    subprocess.call(['gsutil','cp',train_mt_path+'.fam',train_success])
    subprocess.call(['gsutil','cp',test_mt_path+'.fam',test_success])
    if ((not os.path.isfile(train_success) and (get=='both' or get=='train')) 
            or (not os.path.isfile(test_success) and get=='both' or get=='test')):
        mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')
    
        print(f'\nReading UKB phenotype {phen_dict[phen][0]} (code: {phen})...')
        ids0 = hl.import_table('gs://nbaya/risk_gradients/gwas/iid.50.n_324304of324304.seed_1.tsv.bgz',
                               types={'iid':hl.tstr},impute=True)
        ids = ids0.key_by('iid')
        ids = ids.annotate(s = ids.iid)
        ids.describe()
        mt0.describe()
#        mt1 = mt0.annotate_cols(iid = ids[mt0.s].s)
        mt1 = mt0.filter_cols(hl.is_defined(ids[mt0.s]))
        
#        mt1 = mt1.filter_cols(mt1.iid == '',keep=False)

        n_train = mt1.count_cols()
#        n_cas = mt1.filter_cols(mt1.phen == 1).count_cols()
    
        seed = seed if seed is not None else int(str(Env.next_seed())[:8])
#        n_train = int(round(n*(1-test_set)))
#        n_test = n-n_train
#        print('\n###############')
#        print(f'Setting {test_set} of total population to be in the testing set')
#        print(f'n_train = {n_train}\tn_test = {n_test}')
#        print(f'seed = {seed}')
#        print('###############')
#        randstate = np.random.RandomState(int(seed)) #seed random state for replicability
#        labels = ['train']*n_train+['test']*n_test
#        randstate.shuffle(labels)
#        mt2 = mt1.add_col_index('tmp_index').key_cols_by('tmp_index')
#        mt3 = mt2.annotate_cols(set = hl.literal(labels)[hl.int32(mt2.tmp_index)])
#        mt3 = mt3.key_cols_by('s').drop('tmp_index')
        
        
        if (not os.path.isfile(train_success) or overwrite==True) and (get=='train' or get=='both'):
            train_mt = mt1.annotate_cols(phen = 0)
            train_mt, _, _ = downsample(train_mt,frac=1000/n_train,phen=train_mt.phen, for_cases=None,seed=seed)
            n_train = train_mt.count_cols()
            print('\nExporting to PLINK format...')
            hl.export_plink(train_mt,output=train_mt_path,ind_id=train_mt.s,
                            is_female=train_mt.isFemale)

    
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
        header = '\n############\n'
        header += 'Downsampling '+('all' if for_cases is None else ('cases'*for_cases+'controls'*(for_cases==0)))+f' by frac = {frac}\n'
        header += f'n: {n}\n'
        header += f'n_cas: {n_cas}\nn_con: {n-n_cas}\nprevalence: {round(n_cas/n,6)}\n' if for_cases != None else ''
        header += f'seed: {seed}\n'
        header += 'Time: {:%H:%M:%S (%Y-%b-%d)}\n'.format(dt.datetime.now())
        header += '############'
        print(header)        
        randstate = np.random.RandomState(int(seed)) #seed random state for replicability
        col_key  = mt.col_key
        mt = mt.add_col_index('tmp_col_idx')
        mt = mt.key_cols_by('tmp_col_idx')
        for_cases = bool(for_cases) if for_cases != None else None
        filter_arg = (mt[phen_name] == (for_cases==0)) if for_cases != None else (~hl.is_defined(mt[phen_name]))
#        mtA = mt.filter_cols(filter_arg) #keep all individuals in mtA
        mtB = mt.filter_cols(filter_arg , keep=False) #downsample individuals in this mt
        mtB = mtB.add_col_index('col_idx_tmpB')
        mtB = mtB.key_cols_by('col_idx_tmpB')
        nB = n_cas*for_cases + (n-n_cas)*(for_cases==0) if for_cases is not None else n
        n_keep = int(nB*frac)
        labels = ['A']*(n_keep)+['B']*(nB-n_keep)
        randstate.shuffle(labels)
        mtB = mtB.annotate_cols(label = hl.literal(labels)[hl.int32(mtB.col_idx_tmpB)])
        mtB = mtB.filter_cols(mtB.label == 'B') #filter to samples we wish to discard from original mt
        mtB = mtB.key_cols_by('tmp_col_idx')
        mt1 = mt.anti_join_cols(mtB.cols())
        mt1 = mt1.key_cols_by(*col_key)
        n_new = mt1.count_cols()
        n_cas_new = mt1.filter_cols(mt1[phen_name]==1).count_cols()
        elapsed = dt.datetime.now()-start
        print('\n############')
        print('Finished downsampling '+('all' if for_cases is None else ('cases'*for_cases+'controls'*(for_cases==0)))+f' by frac = {frac}')
        print(f'n: {n} -> {n_new} ({round(100*n_new/n,3)}% of original)')
        if n_cas != 0 and n_new != 0 :
            print(f'n_cas: {n_cas} -> {n_cas_new} ({round(100*n_cas_new/n_cas,3)}% of original)')
            print(f'n_con: {n-n_cas} -> {n_new-n_cas_new} ({round(100*(n_new-n_cas_new)/(n-n_cas),3)}% of original)')
            print(f'prevalence: {round(n_cas/n,6)} -> {round(n_cas_new/n_new,6)} ({round(100*(n_cas_new/n_new)/(n_cas/n),6)}% of original)')
        print(f'Time for downsampling: '+str(round(elapsed.seconds/60, 2))+' minutes')
        print('############')
        return mt1, n_new, n_cas_new



if __name__ == "__main__":
    header =  '\n############\n'
    header += f'Phenotypes to downsample: {[phen_dict[phen][0]+f" (code: {phen})" for phen in phen_ls]}\n'
    header += f'Downsampling fractions for all: {frac_all_ls}\n' if frac_all_ls != None else ''
    header += f'Downsampling fractions for cases: {frac_cas_ls}\n' if frac_cas_ls != None else ''
    header += f'Downsampling fractions for controls: {frac_con_ls}\n' if frac_con_ls != None else ''
    header += f'Random seed: {seed}\n'
    header += '############'
    print(header)

#    gt0 = hl.read_matrix_table('gs://phenotype_31063/hail/imputed/ukb31063.GT.autosomes.mt/')
#    gt0 = hl.import_bgen(path='gs://fc-7d5088b4-7673-45b5-95c2-17ae00a04183/imputed/ukb_imp_chr{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22}_v3.bgen',
#                         entry_fields=['GT'],
#                         sample_file = 'gs://ukb31063/ukb31063.autosomes.sample',
#                         variants=variants)
#    variants = hl.import_table('gs://nbaya/split/hapmap3_variants.tsv')
#    variants = variants.annotate(**hl.parse_variant(variants.v))
#    variants = variants.key_by('locus','alleles') 
#    gt0 = hl.import_bgen(path='gs://fc-7d5088b4-7673-45b5-95c2-17ae00a04183/imputed/ukb_imp_chr'+str(set(range(1,23))).replace(' ','')+'_v3.bgen',
#                         entry_fields=['GT'],
#                         n_partitions = 1000,
#                         sample_file = 'gs://ukb31063/ukb31063.autosomes.sample',
#                         variants=variants)


    for phen in phen_ls:
        print('\n############')
        print(f'Starting phenotype: {phen_dict[phen][0]} (code: {phen})')
        print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(dt.datetime.now()))
        print('############')
        mt, n_train, n_cas_train, _, _, _ = get_plink(phen, variant_set, seed=seed, get='both')
