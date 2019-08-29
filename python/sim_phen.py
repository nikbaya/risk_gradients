#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:58:20 2019

Part of risk gradients project.

Simulate phenotype on subset of training set.

@author: nbaya
"""

import hail as hl
import requests
import datetime
import subprocess
import pandas as pd
import argparse
#from hail.experimental.ldscsim import simulate_phenotypes
hl.init(log='/tmp/foo.log')
url = 'https://raw.githubusercontent.com/nikbaya/ldscsim/master/ldscsim.py'
r = requests.get(url).text
exec(r)
calculate_phenotypes = calculate_phenotypes
simulate_phenotypes = simulate_phenotypes


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--modules', type=str, required=True,
                    help="Which modules (numbered) to run")
parser.add_argument('--phi', type=str, required=False,
                    default=None, help="phi for PRS-CS")
args = parser.parse_args()


wd = 'gs://nbaya/risk_gradients/'


def get_subset(maf, variant_set, use_1kg_eur_hm3_snps):

    mt0 = hl.read_matrix_table(
        f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')
    count0 = mt0.count()
    print(f'\n###############\nInitial count: {count0}\n###############')
    withdrawn = hl.import_table(
        'gs://nbaya/w31063_20181016.csv', missing='', no_header=True)
    withdrawn = withdrawn.rename({'f0': 's'})  # rename field with sample IDs
    withdrawn = withdrawn.key_by('s')
    mt1 = mt0.filter_cols(hl.is_defined(withdrawn[mt0.s]), keep=False)
    count1 = mt1.count()
    print(
        f'\n###############\nPost-sample filter count: {count1}\n###############')
    if use_1kg_eur_hm3_snps:
        eur_1kg_hm3_snps = hl.import_table(
            wd+'snpinfo_1kg_hm3.tsv', impute=True)
        eur_1kg_hm3_snps = eur_1kg_hm3_snps.annotate(locus=hl.parse_locus(hl.str(eur_1kg_hm3_snps.CHR)+':'+hl.str(eur_1kg_hm3_snps.BP)),
                                                     alleles=hl.array([eur_1kg_hm3_snps.A1, eur_1kg_hm3_snps.A2]))
        eur_1kg_hm3_snps = eur_1kg_hm3_snps.key_by('locus')
        mt1 = mt1.filter_rows(hl.is_defined(eur_1kg_hm3_snps[mt1.locus]))
        mt1 = mt1.annotate_rows(A1_1kg=eur_1kg_hm3_snps[mt1.locus].A1,
                                A2_1kg=eur_1kg_hm3_snps[mt1.locus].A2)
        mt1 = mt1.filter_rows(((mt1.alleles[0] == mt1.A1_1kg) & (mt1.alleles[1] == mt1.A2_1kg)) | (
            (mt1.alleles[0] == mt1.A2_1kg) & (mt1.alleles[1] == mt1.A1_1kg)))  # keep loci with alleles matching 1kg
        mt1 = mt1.annotate_rows(rsid=eur_1kg_hm3_snps[mt1.locus].SNP)
    mt2 = mt1.annotate_rows(AF=hl.agg.mean(mt1.dosage)/2)
    mt3 = mt2.filter_rows((mt2.AF >= maf) & (mt2.AF <= 1-maf))
    count3 = mt3.count()
    print(
        f'\n###############\nPost-variant filter count: {count3}\n###############')

    return mt3


def check_sim(sim_cols, sim_rows, h2, pi):
    y_no_noise_var = sim_cols.aggregate(hl.agg.stats(sim_cols.y_no_noise)).stdev**2
    print(
        f'\n##############\nExpected h2: {h2}, y_no_noise variance : {y_no_noise_var}\n##############\n')
    y_var = sim_cols.aggregate(hl.agg.stats(sim_cols.y)).stdev**2
    h2_obs = y_no_noise_var/y_var
    print(
        f'\n##############\nExpected h2: {h2}, y variance: {y_var}, Observed h2 (var(y_no_noise)/var(y)): {h2_obs}\n##############\n')
    pi_obs = sim_rows.filter(sim_rows.beta != 0).count()/sim_rows.count()
    print(
        f'\n##############\nExpected pi: {pi}, Observed pi: {pi_obs}\n##############\n')
    return y_no_noise_var, y_var, pi_obs


def gwas_sim_sub(sim, train, n_train_subs, h2, pi, sim_rows, sim_cols, chrom):
    print(f'\n#############\nTraining subsets:\n{n_train_subs}\n#############')

    n_train = train.count()

    # get observed h2 and pi of the full training dataset, 300k individuals
    y_no_noise_var, y_var, pi_obs_full = check_sim(
        sim_rows=sim_rows, sim_cols=sim_cols, h2=h2, pi=pi)
   
    h2_obs_full = y_no_noise_var/y_var

    # for each training subset size
    for n_train_sub in [int(x) for x in n_train_subs]:
        # number of training subsets in full training set with n_train_sub individuals each
        n_subsets = int(n_train/n_train_sub)
        # for (first) training subsets of size n_train_sub
        for subset in range(n_subsets)[:5]:
            gwas_path = wd + \
                f'sim{("" if chrom is "all" else f".chr{chrom}")}.gwas.h2_obs_{round(h2_obs_full,3)}.pi_obs_{round(pi_obs_full,4)}.n_train_sub_{n_train_sub}.subset_{subset}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.v2.tsv.gz'
            try:
                subprocess.check_output([f'gsutil', 'ls', gwas_path]) != None
                print(
                    f'\n#############\nGWAS of subset {subset+1} of {n_subsets} already complete!\n#############')
            except:
                start_sub = datetime.datetime.now()
                train_sub = train.filter(
                    train[f'label_{n_train_sub}'] == str(subset))
                sim_sub = sim.filter_cols(hl.is_defined(train_sub[sim.s]))
                ct = sim_sub.count()
                print(
                    f'\n#############\nExpected # of samples for subset {subset+1} of {n_subsets}: {n_train_sub}, Observed: {ct[1]}\n#############')
                gwas(sim_sub=sim_sub, h2_obs_full=h2_obs_full,
                     pi_obs_full=pi_obs_full, n_train_sub=n_train_sub, 
                     subset=subset, chrom=chrom)
                elapsed_sub = datetime.datetime.now()-start_sub
                print(f'\n#############\nTime for GWAS of subset {subset+1} of {n_subsets}, n = {ct[1]} : '+str(
                    round(elapsed_sub.seconds/60, 2))+' minutes\n#############')


def gwas(sim_sub, h2_obs_full, pi_obs_full, n_train_sub, subset, chrom):
    gwas_path = wd + \
        f'sim{("" if chrom is "all" else f".chr{chrom}")}.gwas.h2_obs_{round(h2_obs_full,3)}.pi_obs_{round(pi_obs_full,4)}.n_train_sub_{n_train_sub}.subset_{subset}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.v2.tsv.gz'
    print(f'\n\nSAVING GWAS RESULTS TO:\n{gwas_path}\n\n')
    cov_list = ['isFemale', 'age', 'age_squared', 'age_isFemale',
                'age_squared_isFemale']+['PC{:}'.format(i) for i in range(1, 21)]
    cov_list = list(
        map(lambda x: sim_sub[x] if type(x) is str else x, cov_list))
    cov_list += [1]

    gwas_sim_sub = hl.linear_regression_rows(y=sim_sub.y,
                                             x=sim_sub.dosage,
                                             covariates=cov_list,
                                             pass_through=['rsid'])
    gwas_sim_sub = gwas_sim_sub.rename(
        {'beta': 'BETA', 'p_value': 'P', 'rsid': 'SNP'})
    gwas_sim_sub = gwas_sim_sub.annotate(A1=gwas_sim_sub.alleles[0],
                                         A2=gwas_sim_sub.alleles[1])
    gwas_sim_sub = gwas_sim_sub.key_by('SNP')
    gwas_sim_sub = gwas_sim_sub.select('A1', 'A2', 'BETA', 'P')
    gwas_sim_sub.export(gwas_path)


def get_corr(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps, phi,chrom):

    mt = hl.read_matrix_table(
        wd+f'genotypes{("" if chrom is "all" else f".chr{chrom}")}.all.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')

    n_train = int(300e3)
    seed = 1
    train = hl.import_table(
        wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')

    test = mt.filter_cols(hl.is_defined(train[mt.s]), keep=False)
    sim_cols = hl.read_table(
        wd+f'sim.cols{("" if chrom is "all" else f".chr{chrom}")}.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.ht')
    test = test.annotate_cols(sim_y=sim_cols[test.s].y)
    test.describe()
    test = test.annotate_rows(A1=test.alleles[0])

    count0 = test.count()
    print(f'\n###############\nInitial count: {count0}\n###############')

    n_test = count0[1]

    n_train_subs = [10e3] #[100e3, 50e3, 20e3,10e3]
    n_train_subs = [int(x) for x in n_train_subs]

    n_train_sub_ls = []
    subset_ls = []
    r_ls = []  # list of correlation coefficients

    for n_train_sub in n_train_subs:
        # number of training subsets in full training set with n_train_sub individuals each
        n_subsets = int(n_train/n_train_sub)
        # for (first) training subsets of size n_train_sub
        for subset in range(n_subsets)[3:6]:
            if n_train_sub != int(50e3) or subset != 1: # temporarily added
                start_sub = datetime.datetime.now()
                path = f'sim.n_train_{int(n_train_sub)}.subset_{subset+1}of{n_subsets}.pst_eff_a1_b0.5_phi{phi}.v2.tsv.gz'
                if chrom!='all':
                    path = f'sim.n_train_sub_{int(n_train_sub)}.subset_{subset}_pst_eff_a1_b0.5_phi1e-04_chr{chrom}.tsv.gz'
                print(f'\n#############\nLoading betas from {path}\n#############')
                betas = hl.import_table(wd+f'prs_cs/{path}',
                                        force=True,
                                        impute=True,
                                        no_header=True)
                betas = betas.rename(
                    {'f0': 'contig', 'f1': 'rsid', 'f2': 'position', 'f3': 'A1', 'f4': 'A2', 'f5': 'adj_beta'})
                betas.show()
                betas = betas.annotate(locus=hl.parse_locus(
                    hl.str(betas.contig)+':'+hl.str(betas.position)))
    
                betas = betas.key_by('locus', 'rsid')
                test1 = test.annotate_rows(adj_beta=betas[test.locus, test.rsid].adj_beta,
                                           prs_A1=betas[test.locus, test.rsid].A1)
    
                test1 = test1.annotate_rows(adj_beta_flipped=(test1.adj_beta*(test1.A1 == test1.prs_A1) +
                                                              -test1.adj_beta*(test1.A1 != test1.prs_A1)))  # flip sign if A1 is not the same in both
    
    #            test1.rows().show(10) #OPTIONAL. Only for checking if flipping was correct
    
                test1 = calculate_phenotypes(
                    mt=test1, genotype=test1.dosage, beta=test1.adj_beta_flipped, h2=1)
                test1 = test1.rename({'y': 'prs'})
    #            stats = test1.aggregate_cols(hl.agg.stats(test1.prs)) #OPTIONAL. Only for checking if prs seems reasonable
    #            print(stats) #OPTIONAL. Only for checking if prs seems reasonable
                r = test1.aggregate_cols(hl.agg.corr(test1.sim_y, test1.prs))
                print(
                    f'\n#############\nr for subset {subset+1} of {n_subsets} (n_train_sub={n_train_sub}): {r}\nr2 for subset {subset+1} of {n_subsets}: {r**2}\n#############')
                n_train_sub_ls.append(n_train_sub)
                subset_ls.append(subset)
                r_ls.append(r)
                elapsed_sub = datetime.datetime.now()-start_sub
                print(f'\n#############\nTime to calculate PRS-phenotype correlation of subset {subset+1} of {n_subsets}, n = {n_train_sub} : {round(elapsed_sub.seconds/60, 2)} minutes\n#############')

    df = pd.DataFrame(data=list(zip(n_train_sub_ls, subset_ls, [n_test]*len(n_train_sub_ls), r_ls)),
                      columns=['n_train', 'subset_id', 'n_test', 'r'])
    print(df)

    hl.Table.from_pandas(df).export(
        wd+f'prs_cs/corr.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.phi_{phi}.v2.tsv')


# get correlation using pruning and thresholding on original betas
def get_corr_pt(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps, chrom, is_pruned,
                is_thresholded):
    # read in testing mt
    mt = hl.read_matrix_table(
        wd+f'genotypes{("" if chrom is "all" else f".chr{chrom}")}.all.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    n_train = int(300e3)
    seed = 1
    train = hl.import_table(
        wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')
    test = mt.filter_cols(hl.is_defined(train[mt.s]), keep=False)
    sim_cols = hl.read_table(
        wd+f'sim.cols{("" if chrom is "all" else f".chr{chrom}")}.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.ht')
    test = test.annotate_cols(sim_y=sim_cols[test.s].y)
    ct_cols = test.count_cols()
    print(f'\n###############\ntest mt col count: {ct_cols}\n###############')

    if is_pruned:
        # define the set of SNPs
        pruned_snps_file = 'ukb_imp_v3_pruned.bim'
        variants = hl.import_table(
            wd+pruned_snps_file, delimiter='\t', no_header=True, impute=True)
        variants = variants.rename(
            {'f0': 'chr', 'f1': 'rsid', 'f3': 'pos'}).key_by('rsid')
        test = test.key_rows_by('rsid')
        # filter to variants defined in variants table
        test = test.filter_rows(hl.is_defined(variants[test.rsid]))
    #    ct_rows = test.count_rows()
    #    print(f'\n###############\ntest mt row count after pruning filter: {ct_rows}\n###############')

    if is_thresholded:
        ss = hl.import_table(wd+f'sim.gwas.h2_obs_0.765.pi_obs_0.001.n_train_sub_100000.subset_0.1kg_eur_hm3.v2.tsv.gz',
                                 impute=True,
                                 force_bgz=True)
    
        ss = ss.key_by('SNP')
        test = test.filter_rows(hl.is_defined(ss[test.rsid]))
    #    ct_rows = test.count_rows()
    #    print(f'\n###############\ntest mt row count after checking against GWAS for 100k: {ct_rows}\n###############')
    
        threshold = 1
    
        print(
            f'\n###############\npval threshold: pval<{threshold}\n###############')
        test = test.annotate_rows(P=ss[test.rsid].P)
        test = test.filter_rows(test.P < threshold)
        ct_rows = test.count_rows()
        print(
            f'\n###############\ntest mt row count after keeping pval<{threshold}: {ct_rows}\n###############')

    n_train_subs = [100e3, 50e3, 20e3, 10e3, 5e3]  # 20e3,
    n_train_subs = [int(x) for x in n_train_subs]

    n_train = 300e3

    for n_train_sub in n_train_subs:
        # number of training subsets in full training set with n_train_sub individuals each
        n_subsets = int(n_train/n_train_sub)
        for subset in range(n_subsets)[:10]:
#            ss_path = wd + \
#                f'sim.gwas.h2_obs_0.765.pi_obs_0.001.n_train_sub_{n_train_sub}.subset_{subset}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.v2.tsv.gz'
            ss_path = wd+f'sim.chr22.gwas.h2_obs_0.485.pi_obs_0.0009.n_train_sub_{n_train_sub}.subset_{subset}.1kg_eur_hm3.v2.tsv.gz'
            print(
                f'\n#############\nn_train_sub={n_train_sub} (subset {subset+1} of {n_subsets})\nss path: {ss_path}\n#############')
            ss = hl.import_table(ss_path,
                                 impute=True,
                                 force=True)
            ss = ss.key_by('SNP')
            test1 = test.annotate_rows(adj_beta=ss[test.rsid].BETA,
                                       prs_A1=ss[test.rsid].A1)
            test1 = test1.annotate_rows(adj_beta_flipped=(test1.adj_beta*(test1.alleles[0] == test1.prs_A1) +
                                                          -test1.adj_beta*(test1.alleles[0] != test1.prs_A1)))  # flip sign if A1 is not the same in both
            test1 = calculate_phenotypes(
                mt=test1, genotype=test1.dosage, beta=test1.adj_beta_flipped, h2=1)
            test1 = test1.rename({'y': 'prs'})
            r = test1.aggregate_cols(hl.agg.corr(test1.sim_y, test1.prs))
            print(
                f'\n#############\nr for subset {subset+1} of {n_subsets} (n_train_sub={n_train_sub}): {r}\nr2 for subset {subset+1} of {n_subsets}: {r**2}\n#############')


def get_corr_ldpred(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps, chrom, 
                    p_and_t, ldpred_inf):
    assert p_and_t != False or ldpred_inf != False, 'p_and_t and ldpred_inf cannot both be False'
    # read in testing mt
    mt = hl.read_matrix_table(
        wd+f'genotypes{("" if chrom is "all" else f".chr{chrom}")}.all.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    n_train = int(300e3)
    seed = 1
    train = hl.import_table(
        wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')
    test = mt.filter_cols(hl.is_defined(train[mt.s]), keep=False)
    sim_cols = hl.read_table(
        wd+f'sim.cols{("" if chrom is "all" else f".chr{chrom}")}.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.ht')
    test = test.annotate_cols(sim_y=sim_cols[test.s].y)
    ct_cols = test.count_cols()
    print(f'\n###############\ntest mt col count: {ct_cols}\n###############')

    n_train_subs = [100e3, 50e3, 20e3, 10e3, 5e3]  # 20e3,
    n_train_subs = [int(x) for x in n_train_subs]

    n_train = 300e3
    
    ii=0 #temporarily used

    for n_train_sub in n_train_subs:
        # number of training subsets in full training set with n_train_sub individuals each
        n_subsets = int(n_train/n_train_sub)
        for subset in range(n_subsets)[:3]:
            if p_and_t:
                
                p_value_thresholds = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
                p_value_thresholds = ['{:.4e}'.format(p) for p in p_value_thresholds]
                
                print(f'\n#############\nStarting P+T for n_train_sub={n_train_sub} (subset {subset+1} of {n_subsets})\nthresholds: {p_value_thresholds}\n#############')
                
                for p_value in p_value_thresholds:
                    ss_path = wd+f'ldpred/sim.chr22.n_train_{n_train_sub}.subset_{subset}.ldr_333_P+T_r0.20_p{p_value}.tsv.gz'
                    print(f'\n#############\nn_train_sub={n_train_sub} (subset {subset+1} of {n_subsets})\nss path: {ss_path}\n#############')
                    ss = hl.import_table(ss_path,
                                         impute=True,
                                         force=True,
                                         delimiter='\t')
                    ss = ss.key_by('sid')
                    print(f'\n#############\nNumber of variants in table (after thresholding): {ss.count()}\nNumber of non-zero betas in table (after pruning): {ss.filter(ss.updated_beta!=0).count()} \n#############')
                    test1 = test.annotate_rows(adj_beta=ss[test.rsid].updated_beta,
                                               ldpred_A1=ss[test.rsid].nt1)
                    test1 = test1.filter_rows(hl.is_defined(test1.adj_beta))
                    test1 = test1.annotate_rows(adj_beta_flipped=(test1.adj_beta*(test1.alleles[0] == test1.ldpred_A1) +
                                                                  -test1.adj_beta*(test1.alleles[0] != test1.ldpred_A1)))  # flip sign if A1 is not the same in both
                    test1 = calculate_phenotypes(
                        mt=test1, genotype=test1.dosage, beta=test1.adj_beta_flipped, h2=1)
                    test1 = test1.rename({'y': 'prs'})
                    r = test1.aggregate_cols(hl.agg.corr(test1.sim_y, test1.prs))
                    print(
                        f'\n#############\nP+T (threshold={p_value})\nr for subset {subset+1} of {n_subsets} (n_train_sub={n_train_sub}): {r}\nr2 for subset {subset+1} of {n_subsets}: {r**2}\n#############')
                    if ii==0:
                        ii+=1
                        test1=test1.annotate_cols(prs_alt = hl.agg.sum(test1.dosage*test1.adj_beta_flipped))
                        r_alt = test1.aggregate_cols(hl.agg.corr(test1.sim_y, test1.prs_alt))
                        print(f'\n#############\nP+T ALT r for subset {subset+1} of {n_subsets} (n_train_sub={n_train_sub}): {r_alt}\nALT r2 for subset {subset+1} of {n_subsets}: {r_alt**2}\n#############')
            if ldpred_inf:
                ss_path = wd+f'ldpred/sim.chr22.n_train_{n_train_sub}.subset_{subset}.ldr_333.h2_0.7.tsv.gz'
                print(f'\n#############\nStarting LDpred-inf for n_train_sub={n_train_sub} (subset {subset+1} of {n_subsets})\nss path: {ss_path}\n#############')
                ss = hl.import_table(ss_path,
                                     impute=True,
                                     force=True)
                ss = ss.key_by('sid')
                test1 = test.annotate_rows(adj_beta=ss[test.rsid].ldpred_inf_beta,
                                           ldpred_A1=ss[test.rsid].nt1)
                test1 = test1.filter_rows(hl.is_defined(test1.adj_beta))
                test1 = test1.annotate_rows(adj_beta_flipped=(test1.adj_beta*(test1.alleles[0] == test1.ldpred_A1) +
                                                              -test1.adj_beta*(test1.alleles[0] != test1.ldpred_A1)))  # flip sign if A1 is not the same in both
                test1 = calculate_phenotypes(
                    mt=test1, genotype=test1.dosage, beta=test1.adj_beta_flipped, h2=1)
                test1 = test1.rename({'y': 'prs'})
                r = test1.aggregate_cols(hl.agg.corr(test1.sim_y, test1.prs))
                print(
                    f'\n#############\nLDpred-inf r for subset {subset+1} of {n_subsets} (n_train_sub={n_train_sub}): {r}\nr2 for subset {subset+1} of {n_subsets}: {r**2}\n#############')

if __name__ == '__main__':

    variant_set = 'qc_pos'
    maf = 0.05
    # whether to subset to the intersection of SNPs in the EUR 1KG SNPs suggested by PRScs
    use_1kg_eur_hm3_snps = True
    h2 = 0.75
    pi = 0.001
#    chrom=22 #use only chromosome 22. default: chrom=all (use all chromosomes)
    chrom='all'
#    exact_h2 = True

    # if modules contains a digit related to an modules, that module will be run
    modules = args.modules
    phi = args.phi  # '1e-02'

    header = '\n###############\n'
    header += f'variant_set: {variant_set}\n'
    header += f'use_1kg_eur_hm3_snps: {use_1kg_eur_hm3_snps}\n'
    header += f'maf: {maf}\n'
    header += f'h2: {h2}\n'
    header += f'pi: {pi}\n'
#    header += f'exact_h2: {exact_h2}\n'
    header += f'phi: {phi}\n' if phi is not None else ''
    header += f'chr: {chrom}\n'
    header += f'modules: {" ".join(sorted(modules))}\n'
    header += '###############'
    print(header)

    file_suffix = f'{("" if chrom is "all" else f".chr{chrom}")}.all.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}'
    sim_suffix = f'{("" if chrom is "all" else f".chr{chrom}")}.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}'

    # Get subset of genotypes (module 1)
    if '1' in modules:
        mt = get_subset(maf, variant_set=variant_set,
                      use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)
        genotype_subset_path = wd+'genotypes'+file_suffix
        print('\nWriting subset of genotypes to {genotype_subset_path}...\n')
        mt.write(genotype_subset_path)

    # Simulate phenotype on subset of genotypes (module 2)
    if '2' in modules:
        print(f'\n#############\nSimulating phenotype (module 2)\n#############')
        start_2 = datetime.datetime.now()
        mt = hl.read_matrix_table(wd+'genotypes'+file_suffix+'.mt')
        if chrom!='all':
            mt = mt.filter_rows(mt.locus.contig==hl.str(chrom))
        print(
            f'\n#############\nSimulating phenotypes on mt with count: {mt.count()}\n#############')
        sim = simulate_phenotypes(mt=mt, genotype=mt.dosage, h2=h2, pi=pi)
        print(
            f"\nWriting simulation cols/rows to:\n{wd+f'sim.cols'+sim_suffix+'.ht'}\n{wd+f'sim.rows'+sim_suffix+'.ht'}\n")
        sim.cols().write(wd+f'sim.cols'+sim_suffix+'.ht',overwrite=True)
        sim.rows().write(wd+f'sim.rows'+sim_suffix+'.ht',overwrite=True)
        elapsed_2 = datetime.datetime.now()-start_2
        print(f'\n#############\nTime to simulate phenotype (module 2) : {round(elapsed_2.seconds/60, 2)} minutes\n#############')

    # Check simulation (module 3)
    if '3' in modules:
        print(f'\n#############\nChecking simulation results (module 3)\n#############')
        sim_cols = hl.read_table(wd+f'sim.cols'+sim_suffix+'.ht')
        sim_rows = hl.read_table(wd+f'sim.rows'+sim_suffix+'.ht')

        y_no_noise_var, y_var, pi_obs = check_sim(sim_cols, sim_rows, h2, pi)

    # Run GWAS on training subsets (module 4)
    if '4' in modules:
        print(f'\n#############\nGWAS on simulated phenotype (module 4)\n#############')
        start_4 = datetime.datetime.now()
        n_train = int(300e3)
        seed = 1
        train = hl.import_table(
            wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')

        # list of number of individuals in each training subset
        n_train_subs = [100e3, 50e3, 20e3, 10e3, 5e3]
        mt = hl.read_matrix_table(wd+'genotypes'+file_suffix+'.mt')
        sim_cols = hl.read_table(wd+f'sim.cols'+sim_suffix+'.ht')
        sim_rows = hl.read_table(wd+f'sim.rows'+sim_suffix+'.ht')
        sim = mt.annotate_cols(y = sim_cols[mt.s].y)
        gwas_sim_sub(sim=sim, train=train, n_train_subs=n_train_subs,
                     h2=h2, pi=pi, sim_rows=sim_rows, sim_cols=sim_cols,
                     chrom=chrom)
        elapsed_4 = datetime.datetime.now()-start_4
        print(f'\n#############\nTime to GWAS simulated phenotype (module 4) : {round(elapsed_4.seconds/60, 2)} minutes\n#############')

    # Get correlation between phenotype and PRS-CS-adjusted PRS (module 5)
    if '5' in modules: 
        print(f'\n#############\nCalculate phenotype-PRS correlation using PRS-CS-derived betas (module 5)\n#############')
        get_corr(variant_set=variant_set, maf=maf, h2=h2, pi=pi,
                 use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps, phi=phi,chrom=chrom)
        
    # Get correlation between phenotype and pruning+thresholding PRS (module 6)
    if '6' in modules:
        print(f'\n#############\nCalculate phenotype-PRS correlation using P+T-derived betas (module 6)\n#############')
        get_corr_pt(variant_set=variant_set, maf=maf, h2=h2, pi=pi,
                    use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps,is_pruned=False,
                    is_thresholded=False,chrom=chrom)

    # Get correlation between phenotype and LDpred PRS (module 7)
    if '7' in modules:
        print(f'\n#############\nCalculate phenotype-PRS correlation using LDpred-derived betas (module 7)\n#############')
        get_corr_ldpred(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps, chrom, 
                    p_and_t=True, ldpred_inf=True)
        
        
        
tb = hl.import_table('/Users/nbaya/Downloads/sim.chr22.n_train_100000.subset_0.ldr_333_P+T_r0.20_p1.0000e-07.tsv',delimiter='    ')        