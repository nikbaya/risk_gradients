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
#from hail.experimental.ldscsim import simulate_phenotypes
url = 'https://raw.githubusercontent.com/nikbaya/ldscsim/master/ldscsim.py'
r = requests.get(url).text
exec(r)

hl.init(log='/tmp/foo.log')

wd = 'gs://nbaya/risk_gradients/'

def check_mt(maf, variant_set, use_1kg_eur_hm3_snps):
    n_train = int(300e3)
    seed = 1
    train = hl.import_table(wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')
    
    
    mt0 = hl.read_matrix_table(f'gs://nbaya/split/ukb31063.{variant_set}_variants.gwas_samples_repart.mt')
    count0 = mt0.count()
    print(f'\n###############\nInitial count: {count0}\n###############')
    mt1 = mt0.filter_cols(hl.is_defined(train[mt0.s]))          
    count1 = mt1.count()
    print(f'\n###############\nPost-sample filter count: {count1}\n###############')
    if use_1kg_eur_hm3_snps:
          eur_1kg_hm3_snps = hl.import_table(wd+'snpinfo_1kg_hm3.tsv',impute=True).key_by('SNP')
          eur_1kg_hm3_snps = eur_1kg_hm3_snps.annotate(locus = hl.parse_locus(hl.str(eur_1kg_hm3_snps.CHR)+hl.str(eur_1kg_hm3_snps.BP)))
          eur_1kg_hm3_snps = eur_1kg_hm3_snps.key_by('locus')
          mt1 = mt1.filter_rows(hl.is_defined(eur_1kg_hm3_snps[mt1.locus]))
    mt2 = mt1.annotate_rows(AF = hl.agg.mean(mt1.dosage)/2)
    mt3 = mt2.filter_rows((mt2.AF>=maf)&(mt2.AF<=1-maf))
#    mt3 = mt1
    count3 = mt3.count()
    print(f'\n###############\nPost-variant filter count: {count3}\n###############')
          
#    variants = hl.read_table(f'gs://nbaya/split/{variant_set}_variants.ht')
#    gt0 = hl.import_bgen(path='gs://fc-7d5088b4-7673-45b5-95c2-17ae00a04183/imputed/ukb_imp_chr'+str(set(range(1,23))).replace(' ','')+'_v3.bgen',
#                         entry_fields=['GT'],
#                         n_partitions = 1000,
#                         sample_file = 'gs://ukb31063/ukb31063.autosomes.sample',
#                         variants=variants)
#    count0 = gt0.count()
#    print(f'\n###############\nInitial count for bgen: {count0}\n###############')
#
#    gt1 = hl.variant_qc(gt0)
#    gt2 = gt1.filter_rows((gt1.variant_qc.AF[0]>=maf)&(gt1.variant_qc.AF[0]<=1-maf))
#    count2 = gt2.count()
#    print(f'\n###############\nPost-variant filter count for bgen: {count2}\n###############')
#          
#    if count1[0] != count2[0]:
#        print(f'\n###############\nWARNING\nmt: {count3}, bgen: {count2}\n###############')
#    
#    
#    mt = gt2
#    mt = mt.filter_cols(hl.is_defined(train[mt.s]))
    
    return mt3

def sim_phen(mt, h2, pi, variant_set, use_1kg_eur_hm3_snps):
    print('\nSimulating phenotypes...')
    sim = simulate_phenotypes(mt=mt, genotype=mt.dosage, h2=h2, pi=pi)
    
    sim.write(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    
def check_sim(sim, h2, pi):
    # heritability
    h2_obs = sim.aggregate_cols(hl.agg.stats(sim.y_no_noise)).stdev**2
    print(f'\n##############\nExpected h2: {h2}, Observed h2: {h2_obs}\n##############\n')
    pi_obs = sim.filter_rows(sim.beta!=0).count_rows()/sim.count_rows()
    print(f'\n##############\nExpected pi: {pi}, Observed pi: {pi_obs}\n##############\n')
    return h2_obs, pi_obs
    
def gwas_sim_sub(sim, train, n_train_subs, h2, pi):
    print(f'\n#############\nTraining subsets:\n{n_train_subs}\n#############')
    
    h2_obs_full, pi_obs_full = check_sim(sim=sim, h2=h2, pi=pi) #get observed h2 and pi of the full training dataset, 300k individuals
    
    for n_train_sub in [int(x) for x in n_train_subs]: #for each training subset size
        n_subsets = int(n_train/n_train_sub) #number of training subsets in full training set with n_train_sub individuals each
        for subset in range(n_subsets)[:1]: #for (first) training subsets of size n_train_sub
            try: 
                subprocess.check_output([f'gsutil','ls',wd+f'sim.gwas.h2_obs_{round(h2_obs_full,3)}.pi_obs_{round(pi_obs_full,4)}.n_train_sub_{n_train_sub}.subset_{subset}.tsv.bgz']) != None
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
    gwas_sim_sub.export(wd+f'sim.gwas.h2_obs_{round(h2_obs_full,3)}.pi_obs_{round(pi_obs_full,4)}.n_train_sub_{n_train_sub}.subset_{subset}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.tsv.bgz')

def get_corr(variant_set, maf, h2, pi, use_1kg_eur_hm3_snps):
    sim = hl.read_matrix_table(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    test = hl.import_table(wd+f'iid.sim.test.n_61144.seed_1.tsv.bgz').key_by('s')
    sim = sim.filter_cols(hl.is_defined(test[sim.s]))
    
    n_train_subs = np.asarray([100e3, 50e3, 20e3, 10e3, 5e3]).astype(int)
    
    for n_train_sub in n_train_subs:
        n_subsets = int(n_train/n_train_sub) #number of training subsets in full training set with n_train_sub individuals each
        for subset in range(n_subsets)[:1]: #for (first) training subsets of size n_train_sub
            betas = hl.import_table(wd+f'sim.n_train_sub_{int(n_train_sub)}.subset_{subset}.pst_eff_a1_b0.5_phiauto.tsv',impute=True, no_header=True)
            betas = betas.rename({'f0':'contig','f1':'rsid','f2':'position','f3':'A1','f4':'A2',})
            betas = betas.key_by('rsid')
            sim1 = sim.annotate_rows(adj_beta = betas[sim.rsid])
            sim1 = sim1.rename({'y':'sim_y','y_no_noise':'sim_y_no_noise'})
            sim1 = calculate_phenotypes(mt=sim1, genotype=sim1.dosage, beta=sim1.adj_beta, h2=1)
            sim1 = sim1.rename({'y':'prs_y','y_no_noise':'prs_y_no_noise'})
            r = sim_sub.aggregate_cols(hl.agg.corr(sim1.sim_y, sim1.prs_y))
            print(f'\n#############\nr for subset {subset+1} of {n_subsets}: {r}\nr2 for subset {subset+1} of {n_subsets}: {r**2}\n#############')
        
    
    


if __name__ == '__main__':
    
    variant_set='qc_pos'
    maf = 0.05
    use_1kg_eur_hm3_snps = True
    h2 = 0.75
    pi = 0.001
    
    print(f'\n###############\nmaf: {maf}\n###############')
          
#    mt = check_mt(maf, variant_set = variant_set, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)
##    
#    mt = mt.checkpoint(wd+f'genotypes.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}.mt')
    
#    gt= hl.read_matrix_table(wd+f'genotypes.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}.mt')
#    
#    eur_1kg_hm3_snps = hl.import_table(wd+'snpinfo_1kg_hm3.tsv',impute=True).key_by('SNP')
#    eur_1kg_hm3_snps = eur_1kg_hm3_snps.annotate(locus = hl.parse_locus(hl.str(eur_1kg_hm3_snps.CHR)+':'+hl.str(eur_1kg_hm3_snps.BP)))
#    eur_1kg_hm3_snps = eur_1kg_hm3_snps.key_by('locus')
#    
#    mt = gt.filter_rows(hl.is_defined(eur_1kg_hm3_snps[gt.locus]))
#    mt = mt.annotate_rows(rsid = eur_1kg_hm3_snps[mt.locus].SNP)
#    
#
#    mt.rsid.show()
#    
#    ct = mt.count()
#    print(f'# of {variant_set} variants with maf > {maf} = {ct[0]}')
#    print(f'# of individuals in training set = {ct[1]}')
##    
#    mt.write(wd+f'genotypes.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    
    
#    mt = hl.read_matrix_table(wd+f'genotypes.{variant_set}.maf_{maf}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
#    
#    sim_phen(mt=mt, h2=h2, pi=pi, variant_set=variant_set, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)
#    
#    sim = hl.read_matrix_table(wd+f'sim.{variant_set}.maf_{maf}.h2_{h2}.pi_{pi}{".1kg_eur_hm3" if use_1kg_eur_hm3_snps else ""}.mt')
    
#    sim.describe()
#    
##    check_sim(sim=sim, h2=h2, pi=pi)
#    
#    n_train = int(300e3)
#    seed = 1
#    train = hl.import_table(wd+f'iid.sim.train.n_{n_train}.seed_{seed}.tsv.bgz').key_by('s')
#    
#
#    n_train_subs = [50e3]#[100e3, 20e3, 10e3, 5e3]#, 50e3, 20e3, 10e3, 5e3] # list of number of individuals in each training subset
#
#    gwas_sim_sub(sim=sim, train=train, n_train_subs=n_train_subs, h2=h2, pi=pi)
    
    get_corr(variant_set=variant_set, h2=h2, pi=pi, variant_set=variant_set, use_1kg_eur_hm3_snps=use_1kg_eur_hm3_snps)
    


