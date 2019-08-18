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
url = 'https://raw.githubusercontent.com/nikbaya/ldscsim/master/ldscsim.py'
r = requests.get(url).text
exec(r)
calculate_phenotypes=calculate_phenotypes
simulate_phenotypes=simulate_phenotypes



parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--phi', type=str, required=False, default=None, help="phi for PRS-CS")
args = parser.parse_args()




hl.init(log='/tmp/foo.log')

wd = 'gs://nbaya/risk_gradients/'

def check_mt(maf, variant_set, use_1kg_eur_hm3_snps):
#    n_train = int(300e3)
#    seed = 1
#    train = hl.import_table(wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')
    
    # Option 1: Read from existing matrix table
    mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')
    count0 = mt0.count()
    print(f'\n###############\nInitial count: {count0}\n###############')
    withdrawn = hl.import_table('gs://nbaya/w31063_20181016.csv',missing='',no_header=True)
    withdrawn = withdrawn.rename({'f0':'s'}) # rename field with sample IDs
    withdrawn = withdrawn.key_by('s')
    mt1 = mt0.filter_cols(hl.is_defined(withdrawn[mt0.s]),keep=False)          
    count1 = mt1.count()
    print(f'\n###############\nPost-sample filter count: {count1}\n###############')
    if use_1kg_eur_hm3_snps:
          eur_1kg_hm3_snps = hl.import_table(wd+'snpinfo_1kg_hm3.tsv',impute=True)
          eur_1kg_hm3_snps = eur_1kg_hm3_snps.annotate(locus = hl.parse_locus(hl.str(eur_1kg_hm3_snps.CHR)+':'+hl.str(eur_1kg_hm3_snps.BP)),
                                                       alleles = hl.array([eur_1kg_hm3_snps.A1,eur_1kg_hm3_snps.A2]))
          eur_1kg_hm3_snps = eur_1kg_hm3_snps.key_by('locus')
          mt1 = mt1.filter_rows(hl.is_defined(eur_1kg_hm3_snps[mt1.locus]))
          mt1 = mt1.annotate_rows(A1_1kg = eur_1kg_hm3_snps[mt1.locus].A1,
                                  A2_1kg = eur_1kg_hm3_snps[mt1.locus].A2)
          mt1 = mt1.filter_rows(((mt1.alleles[0]==mt1.A1_1kg)&(mt1.alleles[1]==mt1.A2_1kg))|(
                  (mt1.alleles[0]==mt1.A2_1kg)&(mt1.alleles[1]==mt1.A1_1kg))) #keep loci with alleles matching 1kg
          mt1 = mt1.annotate_rows(rsid = eur_1kg_hm3_snps[mt1.locus].SNP)
    mt2 = mt1.annotate_rows(AF = hl.agg.mean(mt1.dosage)/2)
    mt3 = mt2.filter_rows((mt2.AF>=maf)&(mt2.AF<=1-maf))
#    mt3 = mt1
    count3 = mt3.count()
    print(f'\n###############\nPost-variant filter count: {count3}\n###############')
    
    return mt3

def sim_phen(mt, h2, pi, variant_set, use_1kg_eur_hm3_snps):
    print(f'\n#############\nSimulating phenotypes on mt with count: {mt.count()}\n#############')
    sim = simulate_phenotypes(mt=mt, genotype=mt.dosage, h2=h2, pi=pi)
    
#    sim.write(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
#    sim.write(wd+f'sim.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    sim.cols().write(wd+f'sim.cols.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.ht')
    sim.rows().write(wd+f'sim.rows.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.ht')
    
def check_sim(sim, sim_cols, sim_rows,h2, pi):
    # heritability
#    h2_obs = sim.aggregate_cols(hl.agg.stats(sim.y_no_noise)).stdev**2
    h2_obs = sim_cols.aggregate(hl.agg.stats(sim_cols.y_no_noise)).stdev**2
    print(f'\n##############\nExpected h2: {h2}, Observed h2: {h2_obs}\n##############\n')
    pi_obs = sim_rows.filter(sim_rows.beta!=0).count()/sim_rows.count()
#    pi_obs = sim.filter_rows(sim.beta!=0).count_rows()/sim.count_rows()
    print(f'\n##############\nExpected pi: {pi}, Observed pi: {pi_obs}\n##############\n')
    return h2_obs, pi_obs
    
def gwas_sim_sub(sim, train, n_train_subs, h2, pi, sim_rows, sim_cols):
    print(f'\n#############\nTraining subsets:\n{n_train_subs}\n#############')
    
    n_train = train.count()
          
    h2_obs_full, pi_obs_full = check_sim(sim=sim, h2=h2, pi=pi,sim_rows=sim_rows,sim_cols=sim_cols) #get observed h2 and pi of the full training dataset, 300k individuals
    
    for n_train_sub in [int(x) for x in n_train_subs]: #for each training subset size
        n_subsets = int(n_train/n_train_sub) #number of training subsets in full training set with n_train_sub individuals each
        for subset in range(n_subsets)[:1]: #for (first) training subsets of size n_train_sub
            gwas_path = wd+f'sim.gwas.h2_obs_{round(h2_obs_full,3)}.pi_obs_{round(pi_obs_full,4)}.n_train_sub_{n_train_sub}.subset_{subset}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.v2.tsv.bgz'
            try: 
                subprocess.check_output([f'gsutil','ls',gwas_path]) != None
                print('\n#############\nGWAS of subset {subset+1} of {n_subsets} already complete!\n#############')
            except:
                start_sub = datetime.datetime.now()
                train_sub = train.filter(train[f'label_{n_train_sub}']==str(subset))
                sim_sub = sim.filter_cols(hl.is_defined(train_sub[sim.s]))
                ct = sim_sub.count()
                print(f'\n#############\nExpected # of samples for subset {subset+1} of {n_subsets}: {n_train_sub}, Observed: {ct[1]}\n#############')
                gwas(sim_sub=sim_sub, h2_obs_full=h2_obs_full, pi_obs_full=pi_obs_full, n_train_sub=n_train_sub, subset=subset)
                elapsed_sub = datetime.datetime.now()-start_sub
                print(f'\n#############\nTime for GWAS of subset {subset+1} of {n_subsets}, n = {ct[1]} : '+str(round(elapsed_sub.seconds/60, 2))+' minutes\n#############')

                  
def gwas(sim_sub, h2_obs_full, pi_obs_full, n_train_sub, subset):
    gwas_path = wd+f'sim.gwas.h2_obs_{round(h2_obs_full,3)}.pi_obs_{round(pi_obs_full,4)}.n_train_sub_{n_train_sub}.subset_{subset}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.v2.tsv.bgz'
    print(f'\n\nSAVING GWAS RESULTS TO:\n{gwas_path}\n\n')
    cov_list = ['isFemale','age','age_squared','age_isFemale',
                        'age_squared_isFemale']+['PC{:}'.format(i) for i in range(1, 21)]
    cov_list = list(map(lambda x: sim_sub[x] if type(x) is str else x, cov_list))
    cov_list += [1]
    
    gwas_sim_sub = hl.linear_regression_rows(y=sim_sub.y,
                                             x=sim_sub.dosage,
                                             covariates=cov_list,
                                             pass_through=['rsid'])
    gwas_sim_sub = gwas_sim_sub.rename({'beta':'BETA','p_value':'P', 'rsid':'SNP'})
    gwas_sim_sub = gwas_sim_sub.annotate(A1 = gwas_sim_sub.alleles[0],
                                         A2 = gwas_sim_sub.alleles[1])
    gwas_sim_sub = gwas_sim_sub.key_by('SNP')
    gwas_sim_sub = gwas_sim_sub.select('A1','A2','BETA','P')
    gwas_sim_sub.export(gwas_path)


def write_test_mt(h2, pi, variant_set, use_1kg_eur_hm3_snps):
    print('\n#############\nWriting out test mt\n#############')
    mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')
    count0 = mt0.count()
    print(f'\n###############\nInitial count: {count0}\n###############')
    test_iids = hl.import_table(wd+f'iid.sim.test.n_61144.seed_1.tsv.bgz').key_by('s')
    test = mt0.filter_cols(hl.is_defined(test_iids[mt0.s]))          
    n_test = test.count_cols()
    print(f'\n###############\nTesting set col count: {n_test}\n###############')
    test = test.annotate_rows(A1 = test.alleles[0])
    eur_1kg_hm3_snps = hl.import_table(wd+'snpinfo_1kg_hm3.tsv',impute=True).key_by('SNP')
    eur_1kg_hm3_snps = eur_1kg_hm3_snps.annotate(locus = hl.parse_locus(hl.str(eur_1kg_hm3_snps.CHR)+':'+hl.str(eur_1kg_hm3_snps.BP)))
    eur_1kg_hm3_snps = eur_1kg_hm3_snps.key_by('locus')
    
    test = test.filter_rows(hl.is_defined(eur_1kg_hm3_snps[test.locus]))
    test = test.annotate_rows(rsid = eur_1kg_hm3_snps[test.locus].SNP)
    
        
    sim = hl.read_matrix_table(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    sim_rows = sim.rows()    
    ct_betas = sim_rows.count()
    print(f'\n#############\nSim row count: {ct_betas}\n#############')
    sim_rows = sim_rows.key_by('rsid','alleles')
    row_ct = test.count_rows()
    print(f'\n#############\nPre-filter row count: {row_ct}\n#############')
    test = test.filter_rows(hl.is_defined(sim_rows[test.rsid,test.alleles]))
    row_ct = test.count_rows()
    print(f'\n#############\nPost-filter row count: {row_ct}\n#############')
          
    test.write(wd+f'genotypes.test.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')

def sim_in_test_mt(h2, pi, variant_set, use_1kg_eur_hm3_snps):
    print('\n#############\nSimulating phenotype for testing set\n#############')
    test = hl.read_matrix_table(wd+f'genotypes.test.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    ct = test.count()
    print(f'\n###############\nInitial count for test mt: {ct}\n###############')

    sim = hl.read_matrix_table(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    sim_rows = sim.rows()    
    ct_betas = sim_rows.count()
    print(f'\n#############\nSim row count: {ct_betas}\n#############')
    sim_rows = sim_rows.key_by('rsid','alleles')
    row_ct = test.count_rows()
    print(f'\n#############\nPre-filter row count: {row_ct}\n#############')
    test = test.filter_rows(hl.is_defined(sim_rows[test.rsid,test.alleles]))
    row_ct = test.count_rows()
    print(f'\n#############\nPost-filter test mt row count: {row_ct}\n#############')
          
    test = test.annotate_rows(beta = sim_rows[test.rsid,test.alleles].beta)
    test_sim = calculate_phenotypes(mt=test, 
                                    genotype=test.dosage, 
                                    beta =test.beta, 
                                    h2=h2)
    test_sim.write(wd+f'sim.test.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    
    
def correct_test_genotypes(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps, phi):
    print('\n#############\nRunning with betas adjusted to genotypes\n#############')

    test_sim = hl.read_matrix_table(wd+f'sim.test.corrected.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    train_sim = hl.read_matrix_table(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    
    train_sim = train_sim.annotate_rows(stats = hl.agg.stats(train_sim.dosage))
    test_sim = test_sim.annotate_rows(train_sim_gt_stats = train_sim.rows()[test_sim.locus,test_sim.alleles].stats)
    test_sim = test_sim.annotate_rows(test_sim_gt_stats = hl.agg.stats(test_sim.dosage))
    
#    test_sim = test_sim.annotate_entries(dosage_adj1 = (test_sim.dosage-test_sim.test_sim_gt_stats.mean)*test_sim.test_sim_gt_stats.stdev)
#    test_sim = test_sim.annotate_entries(dosage_adj2 = (test_sim.dosage_adj1*test_sim.train_sim_gt_stats.stdev)+test_sim.train_sim_gt_stats.mean)

    
    test = test_sim
    
    n_train_subs = [100e3]
    n_train_subs = [int(x) for x in n_train_subs]
    n_train_sub_ls = []
    subset_ls = []
    r_ls = [] #list of correlation coefficients
    n_train=300e3
    for n_train_sub in n_train_subs:
        n_subsets = int(n_train/n_train_sub) #number of training subsets in full training set with n_train_sub individuals each
        for subset in range(n_subsets)[:1]: #for (first) training subsets of size n_train_sub
            print(f'\n#############\nLoading betas from sim.n_train_{int(n_train_sub)}.subset_{subset+1}of{n_subsets}.pst_eff_a1_b0.5_phi{phi}.txt.gz\n#############') 
            betas = hl.import_table(wd+f'prs_cs/sim.n_train_{int(n_train_sub)}.subset_{subset+1}of{n_subsets}.pst_eff_a1_b0.5_phi{phi}.txt.gz',
                                    force=True,
                                    impute=True, 
                                    no_header=True)
            betas = betas.rename({'f0':'contig','f1':'rsid','f2':'position','f3':'A1','f4':'A2','f5':'adj_beta'})
            betas.show()
            betas = betas.annotate(locus = hl.parse_locus(hl.str(betas.contig)+':'+hl.str(betas.position)))
    
            betas = betas.key_by('locus','rsid')
            test1 = test.annotate_rows(adj_beta = betas[test.locus,test.rsid].adj_beta,
                                       prs_A1 = betas[test.locus,test.rsid].A1)

#            test1 = test1.annotate_rows(adj_beta1 = (test1.adj_beta*test1.train_sim_gt_stats.stdev)+test1.train_sim_gt_stats.mean)
#            test1 = test1.annotate_rows(adj_beta2 = (test1.adj_beta1-test1.test_sim_gt_stats.mean)/test1.test_sim_gt_stats.stdev)
#            test1 = test1.annotate_cols(prs = hl.agg.sum(test1.dosage*test1.adj_beta2))
            
            test1 = test1.annotate_entries(train_norm_gt = (test1.dosage-test1.train_sim_gt_stats.mean)/test1.train_sim_gt_stats.stdev)
            test1 = test1.annotate_cols(prs = hl.agg.sum(test1.train_norm_gt*test1.adj_beta))
            
            test1 = test1.rename({'y':'sim_y'})
            r = test1.aggregate_cols(hl.agg.corr(test1.sim_y, test1.prs))
            print(f'\n#############\nr for subset {subset+1} of {n_subsets} (n_train_sub={n_train_sub}): {r}\nr2 for subset {subset+1} of {n_subsets}: {r**2}\n#############')
            n_train_sub_ls.append(n_train_sub)
            subset_ls.append(subset)
            r_ls.append(r)
            
def sim_corrected_test_phen(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps):
    print('\n#############\nSimulating phenotype for testing set\n#############')
    test = hl.read_matrix_table(wd+f'genotypes.test.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    ct = test.count()
    print(f'\n###############\nInitial count for test mt: {ct}\n###############')

    sim = hl.read_matrix_table(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    sim_rows = sim.rows()    
    ct_betas = sim_rows.count()
    print(f'\n#############\nSim row count: {ct_betas}\n#############')
    sim_rows = sim_rows.key_by('rsid','alleles')
    row_ct = test.count_rows()
    print(f'\n#############\nPre-filter row count: {row_ct}\n#############')
    test = test.filter_rows(hl.is_defined(sim_rows[test.rsid,test.alleles]))
    row_ct = test.count_rows()
    print(f'\n#############\nPost-filter test mt row count: {row_ct}\n#############')
    
    test = test.annotate_rows(beta = sim_rows[test.rsid,test.alleles].beta)
          
    train_sim = sim.annotate_rows(stats = hl.agg.stats(sim.dosage))
    test_sim = test.annotate_rows(train_sim_gt_stats = train_sim.rows()[test.locus,test.alleles].stats)
    test_sim = test_sim.annotate_rows(test_sim_gt_stats = hl.agg.stats(test_sim.dosage))
    
    test_sim = test_sim.annotate_entries(train_norm_gt = (test_sim.dosage-test_sim.test_sim_gt_stats.mean)/test_sim.test_sim_gt_stats.stdev)
    test_sim = test_sim.annotate_cols(y_no_noise = hl.agg.sum(test_sim.train_norm_gt*test_sim.beta))
    test_sim = test_sim.annotate_cols(y = test_sim.y_no_noise + hl.rand_norm(0,hl.sqrt(1-h2)))
    
    test_sim.write(wd+f'sim.test.corrected.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')

    
def get_corr(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps, phi):

#    test = hl.read_matrix_table(wd+f'sim.test.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    test = hl.read_matrix_table(wd+f'sim.test.corrected.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    
#    train_sim = hl.read_matrix_table(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
#    n_train = int(300e3)
#    seed = 1
#    train = hl.import_table(wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')
#    subset=1
#    n_train_sub=int(20e3)
#    train_sub = train.filter(train[f'label_{n_train_sub}']==str(subset))
#    train_sim_sub = train_sim.filter_cols(hl.is_defined(train_sub[train_sim.s]))
#    
#    test=train_sim_sub
    
    test.describe()
    test = test.annotate_rows(A1 = test.alleles[0])
    
    count0 = test.count()
    print(f'\n###############\nInitial count: {count0}\n###############')
          
    n_test = count0[1]
        
    n_train_subs = [100e3, 20e3, 5e3] #[100e3, 50e3, 20e3, 
    n_train_subs = [int(x) for x in n_train_subs]
    
    n_train_sub_ls = []
    subset_ls = []
    r_ls = [] #list of correlation coefficients
    
    n_train=300e3
    
    for n_train_sub in n_train_subs:
        n_subsets = int(n_train/n_train_sub) #number of training subsets in full training set with n_train_sub individuals each
        for subset in range(n_subsets)[:1]: #for (first) training subsets of size n_train_sub
            print(f'\n#############\nLoading betas from sim.n_train_{int(n_train_sub)}.subset_{subset+1}of{n_subsets}.pst_eff_a1_b0.5_phi{phi}.txt.gz\n#############') 
            betas = hl.import_table(wd+f'prs_cs/sim.n_train_{int(n_train_sub)}.subset_{subset+1}of{n_subsets}.pst_eff_a1_b0.5_phi{phi}.txt.gz',
                                    force=True,
                                    impute=True, 
                                    no_header=True)
            betas = betas.rename({'f0':'contig','f1':'rsid','f2':'position','f3':'A1','f4':'A2','f5':'adj_beta'})
            betas.show()
            betas = betas.annotate(locus = hl.parse_locus(hl.str(betas.contig)+':'+hl.str(betas.position)))
    
            betas = betas.key_by('locus','rsid')
            test1 = test.annotate_rows(adj_beta = betas[test.locus,test.rsid].adj_beta,
                                       prs_A1 = betas[test.locus,test.rsid].A1)

            test1 = test1.annotate_rows(adj_beta_flipped = (test1.adj_beta*(test1.A1==test1.prs_A1)+
                                                            -test1.adj_beta*(test1.A1!=test1.prs_A1))) #flip sign if A1 is not the same in both
            
            test1.rows().show(10)
            
            test1 = test1.rename({'y':'sim_y','y_no_noise':'sim_y_no_noise'})
            test1 = calculate_phenotypes(mt=test1, genotype=test1.dosage, beta=test1.adj_beta_flipped, h2=1)
            test1 = test1.rename({'y':'prs'})
#            test1 = test1.annotate_cols(prs = hl.agg.sum(test1.dosage*test1.adj_beta))
            stats = test1.aggregate_cols(hl.agg.stats(test1.prs))
            print(stats)
            r = test1.aggregate_cols(hl.agg.corr(test1.sim_y, test1.prs))
            print(f'\n#############\nr for subset {subset+1} of {n_subsets} (n_train_sub={n_train_sub}): {r}\nr2 for subset {subset+1} of {n_subsets}: {r**2}\n#############')
            n_train_sub_ls.append(n_train_sub)
            subset_ls.append(subset)
            r_ls.append(r)
            
    df = pd.DataFrame(data=list(zip(n_train_sub_ls, subset_ls, [n_test]*len(n_train_sub_ls), r_ls)),
                      columns=['n_train','subset_id','n_test','r'])
    print(df)
    
    hl.Table.from_pandas(df).export(wd+f'prs_cs/corr.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.phi_{phi}.tsv')
    

def get_corr_alt(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps, phi):

#    test = hl.read_matrix_table(wd+f'sim.test.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    train_sim = hl.read_matrix_table(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    n_train = int(300e3)
    seed = 1
    train = hl.import_table(wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')
    subset=0
    n_train_sub=int(20e3)
    print(f'\n#############\n# of samples for subset {subset+1} of {int(n_train/n_train_sub)}: {n_train_sub}\n#############')

    train_sub = train.filter(train[f'label_{n_train_sub}']==str(subset))
    train_sim_sub = train_sim.filter_cols(hl.is_defined(train_sub[train_sim.s]))
    test=train_sim_sub
    
    
    test.describe()
    test = test.annotate_rows(A1 = test.alleles[0])
    
    count0 = test.count()
    print(f'\n###############\nInitial count: {count0}\n###############')
          
    n_test = count0[1]
        
    n_train_subs = [5e3]
    n_train_subs = [int(x) for x in n_train_subs]
    
    n_train_sub_ls = []
    subset_ls = []
    r_ls = [] #list of correlation coefficients
    
    n_train=300e3
    
    for n_train_sub in n_train_subs:
        n_subsets = int(n_train/n_train_sub) #number of training subsets in full training set with n_train_sub individuals each
        for subset in range(n_subsets)[:1]: #for (first) training subsets of size n_train_sub
            print(f'\n#############\nLoading betas from sim.n_train_{int(n_train_sub)}.subset_{subset+1}of{n_subsets}.pst_eff_a1_b0.5_phi{phi}.txt.gz\n#############') 
            betas = hl.import_table(wd+f'prs_cs/sim.n_train_{int(n_train_sub)}.subset_{subset+1}of{n_subsets}.pst_eff_a1_b0.5_phi{phi}.txt.gz',
                                    force=True,
                                    impute=True, 
                                    no_header=True)
            betas = betas.rename({'f0':'contig','f1':'rsid','f2':'position','f3':'A1','f4':'A2','f5':'adj_beta'})
            betas.show()
            betas = betas.annotate(locus = hl.parse_locus(hl.str(betas.contig)+':'+hl.str(betas.position)))
    
            betas = betas.key_by('locus','rsid')
            test1 = test.annotate_rows(adj_beta = betas[test.locus,test.rsid].adj_beta,
                                       prs_A1 = betas[test.locus,test.rsid].A1)
            
            
#            betas = hl.import_table(wd+f'sim.gwas.h2_obs_0.766.pi_obs_0.001.n_train_sub_{n_train_sub}.subset_{subset}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.tsv.bgz',
#                                    impute=True,
#                                    force_bgz=True).key_by('SNP')
#            
#            test1 = test.annotate_rows(adj_beta = betas[test.rsid].BETA,
#                                       prs_A1 = betas[test.rsid].A1)            
            
            
            test1 = test1.annotate_rows(adj_beta_flipped = (test1.adj_beta*(test1.A1==test1.prs_A1)+
                                                            -test1.adj_beta*(test1.A1!=test1.prs_A1))) #flip sign if A1 is not the same in both
            
            test1.rows().show(10)
            
            test1 = test1.rename({'y':'sim_y','y_no_noise':'sim_y_no_noise'})
            test1 = calculate_phenotypes(mt=test1, genotype=test1.dosage, beta=test1.adj_beta_flipped, h2=1)
            test1 = test1.rename({'y':'prs'})
#            test1 = test1.annotate_cols(prs = hl.agg.sum(test1.dosage*test1.adj_beta))
            stats = test1.aggregate_cols(hl.agg.stats(test1.prs))
            print(stats)
            r = test1.aggregate_cols(hl.agg.corr(test1.sim_y, test1.prs))
            print(f'\n#############\nr for subset {subset+1} of {n_subsets} (n_train_sub={n_train_sub}): {r}\nr2 for subset {subset+1} of {n_subsets}: {r**2}\n#############')
            n_train_sub_ls.append(n_train_sub)
            subset_ls.append(subset)
            r_ls.append(r)
            
    df = pd.DataFrame(data=list(zip(n_train_sub_ls, subset_ls, [n_test]*len(n_train_sub_ls), r_ls)),
                      columns=['n_train','subset_id','n_test','r'])
    print(df)
    
#    hl.Table.from_pandas(df).export(wd+f'prs_cs/corr.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.phi_{phi}.tsv')
    




if __name__ == '__main__':
    
    variant_set='qc_pos'
    maf = 0.05
    use_1kg_eur_hm3_snps = True #whether to subset to the intersection of SNPs in the EUR 1KG SNPs suggested by PRScs
    h2 = 0.75
    pi = 0.001
    
    phi=args.phi #'1e-02'
    
    print(f'\n###############\nvariant_set: {variant_set}\nuse_1kg_eur_hm3_snps: {use_1kg_eur_hm3_snps}\nmaf: {maf}\n###############')
    
      
#    mt = check_mt(maf, variant_set = variant_set, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)
    
#    mt.write(wd+f'genotypes.all.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    
    mt = hl.read_matrix_table(wd+f'genotypes.all.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
#    
#    sim_phen(mt=mt, h2=h2, pi=pi, variant_set=variant_set, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)
##    
#    sim = hl.read_matrix_table(wd+f'sim.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
#    
#    sim.describe()
    
    sim_cols = hl.read_table(wd+f'sim.cols.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.ht')
    sim_rows = hl.read_table(wd+f'sim.rows.all.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.ht')
    sim = mt.annotate_cols(y = sim_cols[mt.s].y)
        
#    check_sim(sim=sim, h2=h2, pi=pi)
    
    n_train = int(300e3)
    seed = 1
    train = hl.import_table(wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')
    

    n_train_subs = [100e3, 50e3, 20e3, 10e3, 5e3]#, 50e3, 20e3, 10e3, 5e3] # list of number of individuals in each training subset

    gwas_sim_sub(sim=sim, train=train, n_train_subs=n_train_subs, h2=h2, pi=pi, sim_rows=sim_rows, sim_cols=sim_cols)

#    write_test_mt(h2=h2, pi=pi, variant_set=variant_set, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)

    
#    sim_in_test_mt(h2=h2, pi=pi, variant_set=variant_set, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)
    
#    sim_corrected_test_phen(variant_set=variant_set, maf=maf, h2=h2, pi=pi, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)
          
#    get_corr(variant_set=variant_set, maf=maf, h2=h2, pi=pi, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps, phi=phi)
    
#    get_corr_alt(variant_set=variant_set, maf=maf, h2=h2, pi=pi, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps, phi=phi)

#    correct_test_genotypes(variant_set=variant_set, maf=maf, h2=h2, pi=pi, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps, phi=phi)
    
          