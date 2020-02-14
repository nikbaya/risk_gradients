#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb      1 14:50:37 2020

Runs large-scale simulations for testing PRS-CS.

To setup VM:
        conda create -n msprime -y -q python=3.6 numpy=1.18.1 scipy=1.4.1 # create conda environment named msprime and install msprime dependencies
        conda activate msprime # activate msprime environment
        conda install -c -y conda-forge msprime # install msprime Python package

@author: nbaya
"""

import argparse
from datetime import datetime as dt
import numpy as np
from scipy import linalg, random, stats
import math
import msprime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_gwas', default=10000, type=int,
        help='Number of individuals in the discovery GWAS.')
parser.add_argument('--n_test', default=2000, type=int,
        help='Number of individuals in the holdout set for testing PRS.')
parser.add_argument('--n_ref', default=1000, type=int,
        help='Number of individuals in reference panel.')
parser.add_argument('--m_per_chr', default=1000000, type=int,
        help='Length of the region for each chromosome.')
parser.add_argument('--n_chr', default=1, type=int,
        help='Number of chromosomes.')
parser.add_argument('--maf', default=0.05, type=float,
        help='The minor allele frequency cut-off.')
parser.add_argument('--rec', default=2e-8, type=float,
        help='Recombination rate across the region.')
parser.add_argument('--mut', default=2e-8, type=float,
        help='Mutation rate across the region.')
parser.add_argument('--h2_A', default=0.3, type=float,
        help='Additive heritability contribution.')
parser.add_argument('--p_causal', default=1, type=float,
        help='Proportion of SNPs that are causal.')
parser.add_argument('--exact_h2', default=False, action='store_true',
        help='Will set simulated phenotype to have almost exactly the right h2')
parser.add_argument('--rec_map_chr', default=None, type=str,
        help='If you want to pass a recombination map, include the filepath here. '
        'The filename should contain the symbol @, msprimesim will replace instances '
        'of @ with chromosome numbers.')
parser.add_argument('--seed', default=None, type=int,
        help='Seed for replicability. Must be between 1 and (2^32)-1') # random seed is changed for each chromosome when calculating true SNP effects
parser.add_argument('--verbose', '-v', action='store_true', default=False,
                    help='verbose flag')
#parser.add_argument('--out', default='msprime_prs', type=str,
#                help='Output filename prefix.')

def to_log(args, string):
    r'''
    Prints string and sends it to the log file
    '''
    if args is not None:
        logfile =  f'ngwas_{args.n_gwas}.ntest_{args.n_test}.nref_{args.n_ref}.'
        logfile += f'mperchr_{args.m_per_chr}.nchr_{args.n_chr}.h2_{args.h2_A}.'
        logfile += f'pcausal_{args.p_causal}.seed_{args.seed}.log'
        if args.verbose:
            print(string)
    else:
        logfile = None
    if type(string) is not str:
        string = str(string)
    if logfile is not None:
        with open(logfile, 'a') as log:
            log.write(string+'\n')

def get_common_mutations_ts(tree_sequence, maf=0.05, args=None):
#                common_sites = msprime.SiteTable()
#                common_mutations = msprime.MutationTable()

        # Get the mutations > MAF.
        n_haps = tree_sequence.get_sample_size()
        to_log(args=args, string=f'filtering to SNPs w/ MAF>{maf}')

        tables = tree_sequence.dump_tables()
        tables.mutations.clear()
        tables.sites.clear()

        for tree in tree_sequence.trees():
                for site in tree.sites():
                        f = tree.get_num_leaves(site.mutations[0].node) / n_haps # allele frequency
                        if f > maf and f < 1-maf:
                                common_site_id = tables.sites.add_row(
                                        position=site.position,
                                        ancestral_state=site.ancestral_state)
                                tables.mutations.add_row(
                                        site=common_site_id,
                                        node=site.mutations[0].node,
                                        derived_state=site.mutations[0].derived_state)
        new_tree_sequence = tables.tree_sequence()
        return new_tree_sequence

def sim_ts(args):
        r'''
        Simulate tree sequences using out-of-Africa model
        '''
        def initialise(args):
                ts_list = []
                ts_list_geno = []
                genotyped_list_index = []
                m_total, m_geno_total = 0, 0

                m, m_geno, m_start, m_geno_start = [np.zeros(args.n_chr).astype(int) for x in range(4)] # careful not to point to the same object

                return args, ts_list, ts_list_geno, m_total, m_geno_total, m, \
                           m_start, m_geno, m_geno_start, genotyped_list_index

        def out_of_africa(sample_size, no_migration=True):
                N_haps = [2*n for n in sample_size] # double because humans are diploid

                # set population numbers
                N_A = 7300; N_B = 2100
                N_AF = 12300; N_EU0 = 1000; N_AS0 = 510

                # Times are provided in years, so we convert into generations.
                generation_time = 25
                T_AF = 220e3 / generation_time
                T_B = 140e3 / generation_time
                T_EU_AS = 21.2e3 / generation_time

                # We need to work out the starting (diploid) population sizes based on
                # the growth rates provided for these two populations
                r_EU = 0.004
                r_AS = 0.0055
                N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
                N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)

                # Migration rates during the various epochs.
                if no_migration:
                        m_AF_B = 0; m_AF_EU = 0; m_AF_AS = 0; m_EU_AS = 0
                else:
                        m_AF_B = 25e-5; m_AF_EU = 3e-5; m_AF_AS = 1.9e-5; m_EU_AS = 9.6e-5

                # Population IDs correspond to their indexes in the population
                # configuration array. Therefore, we have 0=YRI, 1=CEU and 2=CHB
                # initially.
                n_pops = 3

                pop_configs = [msprime.PopulationConfiguration(sample_size=N_haps[0], initial_size=N_AF),
                                           msprime.PopulationConfiguration(sample_size=N_haps[1], initial_size=N_EU, growth_rate=r_EU),
                                           msprime.PopulationConfiguration(sample_size=N_haps[2], initial_size=N_AS, growth_rate=r_AS)
                                           ]

                migration_mat = [[0, m_AF_EU, m_AF_AS],
                                                 [m_AF_EU, 0, m_EU_AS],
                                                 [m_AF_AS, m_EU_AS, 0],
                                                 ]

                demographic_events = [# CEU and CHB merge into B with rate changes at T_EU_AS
                                                        msprime.MassMigration(time=T_EU_AS, source=2, destination=1, proportion=1.0),
                                                        msprime.MigrationRateChange(time=T_EU_AS, rate=0),
                                                        msprime.MigrationRateChange(time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
                                                        msprime.MigrationRateChange(time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
                                                        msprime.PopulationParametersChange(time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
                                                        # Population B merges into YRI at T_B
                                                        msprime.MassMigration(time=T_B, source=1, destination=0, proportion=1.0),
                                                        # Size changes to N_A at T_AF
                                                        msprime.PopulationParametersChange(time=T_AF, initial_size=N_A, population_id=0)
                                                        ]
                # Return the output required for a simulation study.
                return pop_configs, migration_mat, demographic_events, N_A, n_pops

        # initialize lists
        args, ts_list_all, ts_list_geno_all, m_total, m_geno_total, m, \
        m_start, m_geno, m_geno_start, genotyped_list_index = initialise(args)

        # load recombination maps
        if args.rec_map_chr:
                rec_map_list = []
                for chr_idx in range(args.n_chr):
                        fname = args.rec_map_chr.replace('@',str(chr_idx+1))
                        # TODO: Truncate genetic map to only use m base pairs
                        rec_map_list.append(msprime.RecombinationMap.read_hapmap(fname))
                args.rec, args.m_per_chr = None, None
        else:
                rec_map_list = [None for x in range(args.n_chr)]

        # simulate with out-of-Africa model
        n_total = args.n_gwas + args.n_test + args.n_ref 
        sample_size = [0, n_total, 0] #only set EUR (2nd element in list) sample size to be greater than 0
        pop_configs, migration_mat, demographic_events, Ne, n_pops = out_of_africa(sample_size)

#        dp = msprime.DemographyDebugger(Ne=Ne,
#                                        population_configurations=pop_configs,
#                                        migration_matrix=migration_mat,
#                                        demographic_events=demographic_events)
#        dp.print_history()

        for chr_idx in range(args.n_chr):
                ts_list_all.append(msprime.simulate(sample_size=None, #set to None because sample_size info is stored in pop_configs
                                                    population_configurations=pop_configs,
                                                    migration_matrix=migration_mat,
                                                    demographic_events=demographic_events,
                                                    recombination_map=rec_map_list[chr_idx],
                                                    length=args.m_per_chr, Ne=Ne,
                                                    recombination_rate=args.rec,
                                                    mutation_rate=args.mut,
                                                    random_seed=(args.seed+chr_idx) % 2**32))

                #  get mutations w/ MAF>0
                ts_list_all[chr_idx] = get_common_mutations_ts(ts_list_all[chr_idx], maf=0, args=args) # comment out to run later phenotype simulation with causal SNPs not genotyped

                m[chr_idx] = int(ts_list_all[chr_idx].get_num_mutations())
                m_start[chr_idx] = m_total
                m_total += m[chr_idx]
                to_log(args=args, string=f'number of mutations in chr {chr_idx+1}: {m[chr_idx]}')
                to_log(args=args, string=f'running total of sites : {m_total}')

                ts_list_geno_all.append(ts_list_all[chr_idx])
                genotyped_list_index.append(np.ones(ts_list_all[chr_idx].num_mutations, dtype=bool))
                m_geno[chr_idx] = m[chr_idx]
                m_geno_start[chr_idx] = m_start[chr_idx]
                m_geno_total = m_total
#                to_log(args=args, string=f'Number of sites genotyped in the chr {chr+1}: {m_geno[chr]}')
#                to_log(args=args, string=f'Running total of sites genotyped: {m_geno_total}')


        return args, ts_list_all, ts_list_geno_all, m, m_start, m_total, m_geno, \
                   m_geno_start, m_geno_total, n_pops, genotyped_list_index



def _update_vars(args, ts_list):
        r'''
        update ts_list_geno, genotyped_list_index, m_total, m_geno_total
        '''
        ts_list_geno = []
        genotyped_list_index = []
        m_total = 0
        m_geno_total = 0
        for chr_idx in range(args.n_chr):
                m[chr_idx] = int(ts_list[chr_idx].get_num_mutations())
                m_start[chr_idx] = m_total
                m_total += m[chr_idx]

                ts_list_geno.append(ts_list[chr_idx])
                genotyped_list_index.append(np.ones(ts_list[chr_idx].num_mutations, dtype=bool))
                m_geno[chr_idx] = m[chr_idx]
                m_geno_start[chr_idx] = m_start[chr_idx]
                m_geno_total = m_total
        return ts_list_geno, genotyped_list_index, m_total, m_geno_total

def split(ts_list_both, n1):
        r'''
        split `ts_list_both` into two, with the first half containing the first 
        `n1` samples. 
        '''
        ts_list1 = [ts.simplify(samples=ts.samples()[:2*n1]) for ts in ts_list_both] # first 2*args.n_ref samples in tree sequence are ref individuals
        ts_list2 = [ts.simplify(samples=ts.samples()[2*n1:]) for ts in ts_list_both] # all but first 2*args.n_ref samples in tree sequence are non-ref individuals

        return ts_list1, ts_list2

def nextSNP_add(variant, index=None):
        r'''
        Get normalized genotypes for the given variant. Use `index` to subset to
        desired indiviuals
        '''
        if index is None:
                var_tmp = np.array(variant.genotypes[0::2].astype(int)) + np.array(variant.genotypes[1::2].astype(int))
        else:
                var_tmp = np.array(variant.genotypes[0::2][index].astype(int)) + np.array(variant.genotypes[1::2][index].astype(int))

        # Additive term.
        mean_X = np.mean(var_tmp)
#                p = mean_X / 2
        # Evaluate the mean and then sd to normalise.
        X_A = (var_tmp - mean_X) / np.std(var_tmp)
        return X_A


def sim_phen(args, n_pops, ts_list, m_total):
        r'''
        Simulate phenotype under additive model
        '''
        def set_mutations_in_tree(tree_sequence, p_causal):

                tables = tree_sequence.dump_tables()
                tables.mutations.clear()
                tables.sites.clear()

                causal_bool_index = np.zeros(tree_sequence.num_mutations, dtype=bool)
                # Get the causal mutations.
                for k, site in enumerate(tree_sequence.sites()):
                        if np.random.random_sample() < p_causal:
                                causal_bool_index[k] = True
                                causal_site_id = tables.sites.add_row(
                                        position=site.position,
                                        ancestral_state=site.ancestral_state)
                                tables.mutations.add_row(
                                        site=causal_site_id,
                                        node=site.mutations[0].node,
                                        derived_state=site.mutations[0].derived_state)

                new_tree_sequence = tables.tree_sequence()
                m_causal = new_tree_sequence.num_mutations

                return new_tree_sequence, m_causal, causal_bool_index

        to_log(args=args, string=f'Additive h2 is {args.h2_A}')

        n = int(ts_list[0].get_sample_size()/2 )
        y = np.zeros(n)
        beta_A_list = [] # list of np arrays (one for each chromosome) containing true effect sizes
        ts_pheno_A_list = [] # list of tree sequences on which phenotypes are calculated (possibly ignoring invariant SNPs?)
        causal_A_idx_list = [] # list of booleans indicating if a SNP is causal
        
        np.random.seed(args.seed) # set random seed
        
        m_total = sum([int(ts.get_num_mutations()) for ts in ts_list])
        
        for chr_idx in range(args.n_chr):
                ts = ts_list[chr_idx]
                ts = get_common_mutations_ts(ts, maf=0, args=args)
                to_log(args=args, string=f'picking causal variants and determining effect sizes in chromosome {chr_idx+1}')
                to_log(args=args, string=f'p-causal is {args.p_causal}')
                ts_pheno_A, m_causal_A, causal_A_idx = set_mutations_in_tree(ts, args.p_causal)
                m_chr = int(ts.get_num_mutations())
                to_log(args=args, string=f'picked {m_causal_A} additive causal variants out of {m_chr}')
                beta_A = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal_A)
                beta_A_list.append(beta_A)
                ts_pheno_A_list.append(ts_pheno_A)
                causal_A_idx_list.append(causal_A_idx)

                # additive model for phenotype
                for k, variant in enumerate(ts_pheno_A.variants()): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
                                X_A = nextSNP_add(variant)
                                y += X_A * beta_A[k]

        # add noise to phenotypes
    
        if args.exact_h2:
            y -= np.mean(y)
            y /= np.std(y)
            y *= args.h2_A**(1/2)

            noise = np.random.normal(loc=0, scale=np.sqrt(1-(args.h2_A)), size=n)
            noise -= np.mean(noise)
            noise /= np.std(noise)
            noise *= (1-args.h2_A)**(1/2)
            
            y += noise
        else:
            y += np.random.normal(loc=0, scale=np.sqrt(1-(args.h2_A)), size=n)

        return y, beta_A_list, ts_pheno_A_list, causal_A_idx_list

    
def joint_maf_filter(*ts_lists, args, maf=0.05, logfile=None):
        r'''
        Filter to SNPs with MAF>`maf` in all tree sequence lists passed
        by the `ts_lists`
        '''              
        to_log(args=args, string=f'filtering to SNPs w/ MAF > {args.maf} for {len(ts_lists)} sets of samples')
        for chr_idx in range(args.n_chr):
                ts_dict = {'n_haps': [None for x in range(len(ts_lists))], # dictionary with values = lists (list of lists for sites, positions), each list for a different set of samples
                           'tables': [None for x in range(len(ts_lists))],
                           'sites': [None for x in range(len(ts_lists))],
                           'site_ids': [None for x in range(len(ts_lists))]} 
                    
                for idx, ts_list in enumerate(ts_lists):
                        ts = ts_list[chr_idx]
                        
                        n_haps = ts.get_sample_size()
                        
                        tables = ts.dump_tables()
                        tables.mutations.clear()
                        tables.sites.clear()
                        
                        sites = []
                        for tree in ts.trees():
                                for site in tree.sites():
                                        f = tree.get_num_leaves(site.mutations[0].node) / n_haps # allele frequency
                                        if f > args.maf and f < 1-args.maf:
                                                sites.append(site)
                        site_ids = [(site.position, site.ancestral_state) for site in sites]
                        
                        ts_dict['n_haps'][idx] = n_haps
                        ts_dict['tables'][idx] = tables
                        ts_dict['sites'][idx] = sites
                        ts_dict['site_ids'][idx] = site_ids
                                        
                shared_site_ids = set(ts_dict['site_ids'][0]).intersection(*ts_dict['site_ids'][1:])
                
                ts_dict['sites'] = [[site for site in ts_dict['sites'][idx] \
                                    if (site.position,site.ancestral_state) in shared_site_ids]
                                    for idx,_ in enumerate(ts_lists)]
                
                for idx, _ in enumerate(ts_lists):
                        tables = ts_dict['tables'][idx]
                        
                        for site in ts_dict['sites'][idx]:
                                shared_site = tables.sites.add_row(
                                        position=site.position,
                                        ancestral_state=site.ancestral_state)
                                tables.mutations.add_row(
                                        site=shared_site,
                                        node=site.mutations[0].node,
                                        derived_state=site.mutations[0].derived_state)
                                
                        ts_lists[idx][chr_idx] = tables.tree_sequence()

        return ts_lists
    
    
def get_shared_var_idxs(ts_list1, ts_list2):
        r'''
        Get indices for variants in `ts_list1` that are also in `ts_list2.
        '''
        assert len(ts_list1)==len(ts_list2), 'ts_lists do not have the same length'
        var_idxs_list = []
        for chr_idx, ts1 in enumerate(ts_list1):
                ts2 = ts_list2[chr_idx]
                positions1 = [site.position for tree in ts1.trees() for site in tree.sites()]
                positions2 = [site.position for tree in ts2.trees() for site in tree.sites()]
                var_idxs = [k for k, position in enumerate(positions1) if position in positions2]
                var_idxs = np.asarray(var_idxs)
                var_idxs_list.append(var_idxs)
        return var_idxs_list

def run_gwas(args, y, ts_list_gwas):
        r'''
        Get GWAS beta-hats
        '''
        betahat_A_list = [None for i in range(args.n_chr)] # list of np arrays (one for each chromosome) holding GWAS beta-hats
        maf_A_list = [None for i in range(args.n_chr)] # list of np arrays (one for each chromosome) holding GWAS MAF
#        pval_A_list = [None]*args.n_chr # list of np arrays (one for each chromosome) holding GWAS pvals

        n_gwas = int(ts_list_gwas[0].get_sample_size()/2 ) # assume that sample size is same across chromosomes
        for chr_idx in range(args.n_chr):
                m_gwas = ts_list_gwas[chr_idx].get_num_mutations()
                betahat_A = np.empty(shape=m_gwas)
                maf_A = np.empty(shape=m_gwas)
#                pval_A = np.empty(shape=m_geno_total)
                to_log(args=args, string=f'Determining beta-hats in chromosome {chr_idx+1}')
                for k, variant in enumerate(ts_list_gwas[chr_idx].variants()):
                                X_A = nextSNP_add(variant, index=None)
                                betahat, _, _, pval, _ = stats.linregress(x=X_A, y=y.reshape(n_gwas,))
                                betahat_A[k] = betahat
                                af = variant.genotypes.astype(int).mean()
                                maf = min(af, 1-af)
                                maf_A[k] = maf
#                                pval_A[k] = pval
                betahat_A_list[chr_idx] = betahat_A
                maf_A_list[chr_idx] = maf_A
#                pval_A_list[chr_idx] = pval_A

        return betahat_A_list, maf_A_list
    
def calc_corr(args, causal_idx_pheno_list, causal_idx_list, beta_est_list, 
              y_test, ts_list_test, only_h2_obs=False):
        if not only_h2_obs:
                for chr_idx in range(args.n_chr):
                        causal_idx_pheno = causal_idx_pheno_list[chr_idx]
                        causal_idx = causal_idx_list[chr_idx]
                        if len(causal_idx_pheno)==0 or len(causal_idx)==0:
                                break
                        beta_est = np.squeeze(beta_est_list[chr_idx])
                        beta_A_pheno = np.zeros(shape=len(beta_est))
                        beta_A_pheno[causal_idx] = beta_A_list[chr_idx][causal_idx_pheno]
                        r = np.corrcoef(np.vstack((beta_A_pheno, beta_est)))[0,1]
                        to_log(args=args, string=f'correlation between betas: {r}')
    
        n = int(ts_list_test[0].get_sample_size()/2 )
        yhat = np.zeros(n)
        for chr_idx in range(args.n_chr):
                ts_geno = ts_list_test[chr_idx]
                beta_est = np.squeeze(beta_est_list[chr_idx])
                m_geno = len([x for x in ts_geno.variants()])
                if len(beta_est) < m_geno:
                        beta_est0 = beta_est.copy()
                        causal_idx_pheno = causal_idx_pheno_list[chr_idx]
                        causal_idx = causal_idx_list[chr_idx]
                        beta_est = np.zeros(shape=m_geno)
                        beta_est[causal_idx] = beta_est0[causal_idx_pheno]
                for k, variant in enumerate(ts_geno.variants()): 
                        X_A = nextSNP_add(variant)
                        yhat += X_A * beta_est[k]
        r = np.corrcoef(np.vstack((y_test, yhat)))[0,1]
        if only_h2_obs:
            to_log(args=args, string=f'h2 obs. (y w/ y_gen R^2): {r**2}')
        else:
            to_log(args=args, string=f'y w/ yhat correlation: {r}')


def calc_ld(args, ts_list_ref):
        r'''
        Calculate LD for reference panel
        '''
        ld_list = []
        for chr_idx in range(args.n_chr):
                X = ts_list_ref[chr_idx].genotype_matrix()
                ld = np.corrcoef(X)
                ld_list.append([ld])
        return ld_list

def prs_cs(args, betahat_A_list, maf_A_list, ld_list):
        r'''
        Use PRS-CS to calculate adjusted beta-hats
        '''
        def _psi(x, alpha, lam):
                f = -alpha*(math.cosh(x)-1)-lam*(math.exp(x)-x-1)
                return f

        def _dpsi(x, alpha, lam):
                f = -alpha*math.sinh(x)-lam*(math.exp(x)-1)
                return f

        def _g(x, sd, td, f1, f2):
                if (x >= -sd) and (x <= td):
                        f = 1
                elif x > td:
                        f = f1
                elif x < -sd:
                        f = f2

                return f

        def gigrnd(p, a, b):
                # setup -- sample from the two-parameter version gig(lam,omega)
                p = float(p); a = float(a); b = float(b)
                lam = p
                omega = math.sqrt(a*b)

                if lam < 0:
                        lam = -lam
                        swap = True
                else:
                        swap = False

                alpha = math.sqrt(math.pow(omega,2)+math.pow(lam,2))-lam

                # find t
                x = -_psi(1, alpha, lam)
                if (x >= 1/2) and (x <= 2):
                        t = 1
                elif x > 2:
                        t = math.sqrt(2/(alpha+lam))
                elif x < 1/2:
                        t = math.log(4/(alpha+2*lam))

                # find s
                x = -_psi(-1, alpha, lam)
                if (x >= 1/2) and (x <= 2):
                        s = 1
                elif x > 2:
                        s = math.sqrt(4/(alpha*math.cosh(1)+lam))
                elif x < 1/2:
                        s = min(1/lam, math.log(1+1/alpha+math.sqrt(1/math.pow(alpha,2)+2/alpha)))

                # find auxiliary parameters
                eta = -_psi(t, alpha, lam)
                zeta = -_dpsi(t, alpha, lam)
                theta = -_psi(-s, alpha, lam)
                xi = _dpsi(-s, alpha, lam)

                p = 1/xi
                r = 1/zeta

                td = t-r*eta
                sd = s-p*theta
                q = td+sd

                # random variate generation
                while True:
                        U = random.random()
                        V = random.random()
                        W = random.random()
                        if U < q/(p+q+r):
                                rnd = -sd+q*V
                        elif U < (q+r)/(p+q+r):
                                rnd = td-r*math.log(V)
                        else:
                                rnd = -sd+p*math.log(V)

                        f1 = math.exp(-eta-zeta*(rnd-t))
                        f2 = math.exp(-theta+xi*(rnd+s))
                        if W*_g(rnd, sd, td, f1, f2) <= math.exp(_psi(rnd, alpha, lam)):
                                break

                # transform back to the three-parameter version gig(p,a,b)
                rnd = math.exp(rnd)*(lam/omega+math.sqrt(1+math.pow(lam,2)/math.pow(omega,2)))
                if swap:
                        rnd = 1/rnd

                rnd = rnd/math.sqrt(a/b)
                return rnd

        def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin,
                 chrom, beta_std, seed, args):
                to_log(args=args, string=f'... MCMC (chr{chrom})...')

                # seed
                if seed != None:
                        random.seed(seed)

                # derived stats
                beta_mrg = np.array(sst_dict['BETA']).T
                beta_mrg = np.expand_dims(beta_mrg, axis=1)
                maf = np.array(sst_dict['MAF']).T
                n_pst = (n_iter-n_burnin)/thin
#                p = len(sst_dict['SNP'])
                p = len(sst_dict['BETA'])
                n_blk = len(ld_blk)

                # initialization
                beta = np.zeros((p,1))
                psi = np.ones((p,1))
                sigma = 1.0
                if phi == None:
                        phi = 1.0; phi_updt = True
                else:
                        phi_updt = False

                beta_est = np.zeros((p,1))
                psi_est = np.zeros((p,1))
                sigma_est = 0.0
                phi_est = 0.0

                # MCMC
                for itr in range(1,n_iter+1):
#                        if itr % 100 == 0:
#                                to_log(args=args, string='--- iter-' + str(itr) + ' ---')

                        mm = 0; quad = 0.0
                        for kk in range(n_blk):
                                if blk_size[kk] == 0:
                                        continue
                                else:
                                        idx_blk = range(mm,mm+blk_size[kk])
                                        dinvt = ld_blk[kk]+np.diag(1.0/psi[idx_blk].T[0])
                                        dinvt_chol = linalg.cholesky(dinvt)
                                        beta_tmp = linalg.solve_triangular(dinvt_chol, beta_mrg[idx_blk], trans='T') + np.sqrt(sigma/n)*random.randn(len(idx_blk),1)
                                        beta[idx_blk] = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N')
                                        quad += np.dot(np.dot(beta[idx_blk].T, dinvt), beta[idx_blk])
                                        mm += blk_size[kk]

                        err = max(n/2.0*(1.0-2.0*sum(beta*beta_mrg)+quad), n/2.0*sum(beta**2/psi))
                        sigma = 1.0/random.gamma((n+p)/2.0, 1.0/err)

                        delta = random.gamma(a+b, 1.0/(psi+phi))

                        for jj in range(p):
                                psi[jj] = gigrnd(a-0.5, 2.0*delta[jj], n*beta[jj]**2/sigma)
                        psi[psi>1] = 1.0

                        if phi_updt == True:
                                w = random.gamma(1.0, 1.0/(phi+1.0))
                                phi = random.gamma(p*b+0.5, 1.0/(sum(delta)+w))

                        # posterior
                        if (itr>n_burnin) and (itr % thin == 0):
                                beta_est = beta_est + beta/n_pst
                                psi_est = psi_est + psi/n_pst
                                sigma_est = sigma_est + sigma/n_pst
                                phi_est = phi_est + phi/n_pst

                # convert standardized beta to per-allele beta
                if beta_std == 'False':
                        beta_est /= np.sqrt(2.0*maf*(1.0-maf))

#                # write posterior effect sizes
#                if phi_updt == True:
#                        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
#                else:
#                        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)
#
#                with open(eff_file, 'w') as ff:
#                        for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
#                                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta))

                # print estimated phi
#                if phi_updt == True:
#                        to_log(args=args, string='... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

#                to_log(args=args, string='... Done ...')
                return beta_est

        a = 1; b = 0.5
        phi = None
        n = args.n_gwas
        n_iter = 1000
        n_burnin = 500
        thin = 5
        beta_std = True
        seed = args.seed

        sst_dict_list = [{'BETA':betahat_A_list[chr_idx], 'MAF':maf_A_list[chr_idx]}
                                         for chr_idx in range(args.n_chr)]

        beta_est_list = []
        for chr_idx in range(args.n_chr):
                sst_dict = sst_dict_list[chr_idx]
                ld_blk = ld_list[chr_idx]
                blk_size = [len(blk) for blk in ld_blk]
                beta_est = mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size,
                                  n_iter, n_burnin, thin, chr_idx+1, 
                                  beta_std, seed, args=args)
                beta_est_list.append(beta_est)
        return beta_est_list

if __name__ == '__main__':
        args = parser.parse_args()
        
        to_log(args=args, string=f'start time: {dt.now().strftime("%d/%m/%Y %H:%M:%S")}')
        to_log(args=args, string=args)
        
        # TODO: Consider adding argument for genotype proportion

        # simulate tree sequences
        start_sim_ts = dt.now()
        args, ts_list_all, ts_list_geno_all, m, m_start, m_total, m_geno, m_geno_start, \
        m_geno_total, n_pops, genotyped_list_index = sim_ts(args=args)
        to_log(args=args, string=f'sim ts time (min): {round((dt.now()-start_sim_ts).seconds/60, 2)}\n')

        # split into ref and non-ref (non-ref will have both the gwas and test sets)
        ts_list_ref, ts_list_nonref = split(ts_list_both=ts_list_all, 
                                            n1=args.n_ref)

        # simulate phenotype
        start_sim_phen = dt.now()
        y, beta_A_list, ts_pheno_A_list, causal_A_idx_list = sim_phen(args=args, 
                                                                      n_pops=n_pops, 
                                                                      ts_list=ts_list_nonref, 
                                                                      m_total=m_total)
        assert y.shape[0] == args.n_gwas+args.n_test
        y_gwas = y[:args.n_gwas] # take first n_gwas individuals, just like in the splitting of ts_list_nonref
        y_test = y[args.n_gwas:] # take the complement of the first n_gwas individuals, just like in the splitting of ts_list_nonref
        # TODO: Check that individuals are in the same order in ts_pheno_A_list and ts_list_nonref
        to_log(args=args, string=f'sim phen time: {round((dt.now()-start_sim_phen).seconds/60, 2)} min\n')
        
        # split non-ref into gwas and test sets
        ts_list_gwas, ts_list_test = split(ts_list_both=ts_list_nonref, 
                                           n1=args.n_gwas) 

        # MAF filter ref, gwas, and test cohorts
        # TODO: remove necessity of passing args to this function to get n_chr
        start_joint_maf = dt.now()
        ts_list_ref, ts_list_gwas, ts_list_test = joint_maf_filter(ts_list_ref, 
                                                                   ts_list_gwas,
                                                                   ts_list_test,
                                                                   args=args,
                                                                   maf=args.maf)
        # get causal variant indices for the GWAS cohort
        causal_idx_pheno_gwas_list = get_shared_var_idxs(ts_pheno_A_list, ts_list_nonref)
        causal_idx_gwas_pheno_list = get_shared_var_idxs(ts_list_nonref, ts_pheno_A_list) 
                
        # TODO: calculate observed h2
        calc_corr(args=args, 
                  causal_idx_pheno_list=causal_idx_pheno_gwas_list, 
                  causal_idx_list=causal_idx_gwas_pheno_list,
                  beta_est_list=beta_A_list,
                  y_test=y,
                  ts_list_test=ts_list_nonref,
                  only_h2_obs=True)
        
        # TODO: update _update_vars to remove extraneous code
        _, genotyped_list_index, m_total, m_geno_total = _update_vars(args=args, 
                                                                      ts_list=ts_list_gwas) # update only for discovery cohort
        to_log(args=args, string=f'\tpost maf filter variant ct: {m_total}')
        to_log(args=args, string=f'joint maf filter time: {round((dt.now()-start_joint_maf).seconds/60, 2)} min\n')


        # get causal variant indices for the test cohort
        causal_idx_pheno_list = get_shared_var_idxs(ts_pheno_A_list, ts_list_test)
        causal_idx_test_list = get_shared_var_idxs(ts_list_test, ts_pheno_A_list)

        # run GWAS (and calculate MAF along the way)
        # TODO: Make sure that y_gwas corresponds to the right individuals
        start_run_gwas = dt.now()
        betahat_A_list, maf_A_list = run_gwas(args=args, 
                                              y=y_gwas, 
                                              ts_list_gwas=ts_list_gwas)
        to_log(args=args, string=f'run gwas time: {round((dt.now()-start_run_gwas).seconds/60, 2)} min\n')

        # calculate beta/betahat and y/yhat correlations
        calc_corr(args=args, 
                  causal_idx_pheno_list=causal_idx_pheno_list, 
                  causal_idx_list=causal_idx_test_list, 
                  beta_est_list=betahat_A_list,
                  y_test=y_test,
                  ts_list_test=ts_list_test)

        # calculate LD matrix
        start_calc_ld = dt.now()
        ld_list = calc_ld(args=args,
                          ts_list_ref=ts_list_ref)
        to_log(args=args, string=f'calc ld time: {round((dt.now()-start_calc_ld).seconds/60, 2)} min\n')

        # run PRS-CS
        start_prs_cs = dt.now()
        beta_est_list = prs_cs(args=args, 
                               betahat_A_list=betahat_A_list, 
                               maf_A_list=maf_A_list, 
                               ld_list=ld_list)
        to_log(args=args, string=f'prs-cs time: {round((dt.now()-start_prs_cs).seconds/60, 2)} min\n')


        # calculate beta/betahat and y/yhat correlations
        calc_corr(args=args, 
                  causal_idx_pheno_list=causal_idx_pheno_list, 
                  causal_idx_list=causal_idx_test_list, 
                  beta_est_list=beta_est_list, 
                  y_test=y_test,
                  ts_list_test=ts_list_test)

        to_log(args=args, string=f'total time (min): {round((dt.now()-start_sim_ts).seconds/60, 2)}')
        

