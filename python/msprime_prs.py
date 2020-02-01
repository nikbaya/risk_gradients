#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:50:37 2020

Runs large-scale simulations for testing PRS-CS

@author: nbaya
"""

import argparse
from datetime import datetime as dt
import numpy as np
import math
import msprime

parser = argparse.ArgumentParser()
parser.add_argument('--n', default=10000, type=int,
    help='Number of individuals in the sampled genotypes [Default: 10,000].')
parser.add_argument('--n_ref', default=1000, type=int,
    help='Number of individuals in reference panel [Default: 1,000].')
parser.add_argument('--m_per_chr', default=1000000, type=int,
    help='Length of the region for each chromosome [Default: 1,000,000].')
parser.add_argument('--n_chr', default=1, type=int,
    help='Number of chromosomes [Default: 1].')
parser.add_argument('--maf', default=0.05, type=float,
    help='The minor allele frequency cut-off [Default: 0.05].')
parser.add_argument('--rec', default=2e-8, type=float,
    help='Recombination rate across the region [Default: 2e-8].')
parser.add_argument('--mut', default=2e-8, type=float,
    help='Mutation rate across the region [Default: 2e-8].')
parser.add_argument('--h2_A', default=0.3, type=float,
    help='Additive heritability contribution [Default: 0.3].')
parser.add_argument('--p_causal', default=1, type=float,
    help='Proportion of SNPs that are causal [Default: 1].')
parser.add_argument('--rec_map_chr', default=None, type=str,
    help='If you want to pass a recombination map, include the filepath here. '
    'The filename should contain the symbol @, msprimesim will replace instances '
    'of @ with chromosome numbers.')
parser.add_argument('--out', default='msprime_prs', type=str,
        help='Output filename prefix.')


def sim_ts(args):
    r'''
    Simulate tree sequences using out-of-Africa model
    '''
    def initialise(args):
        ts_list = []
        ts_list_geno = []
        genotyped_list_index = []
        m_total, m_geno_total = 0, 0
    
        if args.load_tree_sequence is not None:
            args.n_chr = 1
    
        m = m_geno = m_start = m_geno_start = np.zeros(args.n_chr).astype(int)
    
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

    def get_common_mutations_ts(args, tree_sequence):
#        common_sites = msprime.SiteTable()
#        common_mutations = msprime.MutationTable()
    
        # Get the mutations > MAF.
        n_haps = tree_sequence.get_sample_size()
        print(f'Determining sites > MAF cutoff {args.maf}')
    
        tables = tree_sequence.dump_tables()
        tables.mutations.clear()
        tables.sites.clear()
    
        for tree in tree_sequence.trees():
            for site in tree.sites():
                f = tree.get_num_leaves(site.mutations[0].node) / n_haps
                if f > args.maf and f < 1-args.maf:
                    common_site_id = tables.sites.add_row(
                        position=site.position,
                        ancestral_state=site.ancestral_state)
                    tables.mutations.add_row(
                        site=common_site_id,
                        node=site.mutations[0].node,
                        derived_state=site.mutations[0].derived_state)
        new_tree_sequence = tables.tree_sequence()
        return new_tree_sequence


    # initialize lists
    args, ts_list, ts_list_geno, m_total, m_geno_total, m, \
    m_start, m_geno, m_geno_start, genotyped_list_index = initialise(args)
    
    # load recombination maps
    if args.rec_map_chr:
        pass
#        rec_map_list = []
#        for chr in range(args.n_chr):
#            rec_map_list.append(msprime.RecombinationMap.read_hapmap(tl.sub_chr(args.rec_map_chr, chr+1)))
#        args.rec, args.m = None, None
    else:
        rec_map_list = [None]*args.n_chr
    
    # simulate with out-of-Africa model
    print(f'... simulating out-of-Africa model for {args.n} EUR samples ...')
    sample_size = [0, args.n, 0] #only set EUR sample size to be greater than 0
    pop_configs, migration_mat, demographic_events, Ne, n_pops = out_of_africa(sample_size)
    
    dp = msprime.DemographyDebugger(Ne=args.Ne,
                                    population_configurations=pop_configs,
                                    migration_matrix=migration_mat,
                                    demographic_events=demographic_events)
    dp.print_history()
    
    for chr in range(args.n_chr):
        ts_list.append(msprime.simulate(sample_size=sample_size,
                                        configurations=pop_configs,
                                        migration_matrix=migration_mat,
                                        demographic_events=demographic_events,
                                        recombination_map=rec_map_list[chr],
                                        length=args.m_per_chr, Ne=Ne, 
                                        recombination_rate=args.rec, 
                                        mutation_rate=args.mut))
        # assign betas
#        common_mutations = []
#        N_haps_chr = ts_list[chr].get_sample_size()
        
        # get mutations > MAF
        ts_list[chr] = get_common_mutations_ts(args, ts_list[chr])
        
        m[chr] = int(ts_list[chr].get_num_mutations())
        m_start[chr] = m_total
        m_total += m[chr]
        print(f'Number of mutations above MAF in the generated data: {m[chr]}')
        print('Running total of sites > MAF cutoff: {m_total}')
        
        ts_list_geno.append(ts_list[chr])
        genotyped_list_index.append(np.ones(ts_list[chr].num_mutations, dtype=bool))
        m_geno[chr] = m[chr]
        m_geno_start[chr] = m_start[chr]
        m_geno_total = m_total
        print('Number of sites genotyped in the generated data: {m_geno[chr]}')
        print('Running total of sites genotyped: {m_geno_total}')

    
    return ts_list, ts_list_geno, m, m_start, m_total, m_geno, m_geno_start, \
           m_geno_total, n_pops, genotyped_list_index

def nextSNP_add(variant, index=None):
    	if index is None:
    		var_tmp = np.array(variant.genotypes[0::2].astype(int)) + np.array(variant.genotypes[1::2].astype(int))
    	else:
    		var_tmp = np.array(variant.genotypes[0::2][index].astype(int)) + np.array(variant.genotypes[1::2][index].astype(int))
    
    	# Additive term.
    	mean_X = np.mean(var_tmp)
#    	p = mean_X / 2
    	# Evaluate the mean and then sd to normalise.
    	X_A = (var_tmp - mean_X) / np.std(var_tmp)
    	return X_A


def sim_phen(args, n_pops, ts_list, m_total):
    '''
    Simulate phenotype under additive model
    '''
    def set_mutations_in_tree(tree_sequence, p_causal):

    	tables = tree_sequence.dump_tables()
    	tables.mutations.clear()
    	tables.sites.clear()
    	
    	causal_bool_index = np.zeros(tree_sequence.num_mutations, dtype=bool)
    	# Get the causal mutations.
    	k = 0
    	for site in tree_sequence.sites():
    		if np.random.random_sample() < p_causal:
    			causal_bool_index[k] = True
    			causal_site_id = tables.sites.add_row(
    				position=site.position,
    				ancestral_state=site.ancestral_state)
    			tables.mutations.add_row(
    				site=causal_site_id,
    				node=site.mutations[0].node,
    				derived_state=site.mutations[0].derived_state)
    		k = k+1
    
    	new_tree_sequence = tables.tree_sequence()
    	m_causal = new_tree_sequence.num_mutations
    
    	return new_tree_sequence, m_causal, causal_bool_index
    
    print(f'Additive h2 is {args.h2_A}')
    
    y = np.zeros(args.n)
    beta_A_list = [] # list of np arrays (one for each chromosome) containing true effect sizes
    
    for chr in range(args.n_chr):
        m_chr = int(ts_list[chr].get_num_mutations())
        print(f'Picking causal variants and determining effect sizes in chromosome {chr+1}')
        print(f'p-causal is {args.p_causal}')
        ts_pheno_A, m_causal_A, causal_A_index = set_mutations_in_tree(ts_list[chr], args.p_causal)
        print(f'Picked {m_causal_A} additive causal variants out of {m_chr}')
        beta_A = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal_A)
        beta_A_list.append(beta_A)
        
        # Get the phenotypes.
        for k, variant in enumerate(ts_pheno_A.variants()): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
                X_A = nextSNP_add(variant)
                y += X_A * beta_A[k]

    return y, beta_A_list
    
    
def run_gwas(args, y, ts_list_geno, m_geno_total):
    r'''
    Get GWAS beta-hats
    '''
    intercept = np.ones(shape=(1,args.n)) # vector of intercepts for least sq linear regression
    betahat_A_list = [None]*args.n_chr # list of np arrays (one for each chromosome) holding GWAS beta-hats
    
    # TODO: Check that betas for intercept term are zero because X_A is normalized
    for chr in range(args.n_chr):
            betahat_A = np.empty(shape=m_geno_total)
            print('Determining beta-hats in chromosome {chr+1}')
            for k, variant in enumerate(ts_list_geno[chr].variants()):
                    X_A = nextSNP_add(variant, index=None)
                    X_A_w_int = np.vstack((X_A.reshape(1, args.n), intercept)).T
                    coef, _, _, _ = np.linalg.lstsq(X_A_w_int, y.reshape(args.n,),rcond=None)
                    betahat_A[k] = coef[0] # take only the beta for the genotypes, not the intercept
            betahat_A_list[chr] = betahat_A
    
    return betahat_A_list
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    n_total = args.n + args.n_ref
    
    # simulate tree sequences
    start_sim_ts = dt.now()
    ts_list, ts_list_geno, m, m_start, m_total, m_geno, m_geno_start, \
    m_geno_total, n_pops, genotyped_list_index  = sim_ts(args, n_total)
    print(f'sim ts time (min): {round((dt.now()-start_sim_ts).seconds/60, 2)}')
    
    # TODO: Split into ref and non-ref, MAF filter, then take the intersection of SNPs passing MAF filter
    ts_list_ref = [ts.simplify(samples=ts.samples()[:args.n_ref]) for ts in ts_list] # first args.n_ref samples in tree sequence are ref individuals
    ts_list_tar = [ts.simplify(samples=ts.samples()[args.n_ref:]) for ts in ts_list] # all but first args.n_ref samples in tree sequence are target individuals

    ts_list_geno_ref = [ts.simplify(samples=ts.samples()[:args.n_ref]) for ts in ts_list_geno] # first args.n_ref samples in tree sequence are ref individuals
    ts_list_geno_tar = [ts.simplify(samples=ts.samples()[args.n_ref:]) for ts in ts_list_geno] # all but first args.n_ref samples in tree sequence are target individuals
    
    
    
    # simulate phenotype
    start_sim_phen = dt.now()
    y, beta_A_list = sim_phen(args, n_pops, ts_list, m_total)
    print(f'sim phen time (min): {round((dt.now()-start_sim_phen).seconds/60, 2)}')

    # run GWAS
    start_run_gwas = dt.now()
    betahat_A_list = run_gwas(args, y, ts_list_geno, m_geno_total)
    print(f'run gwas time (min): {round((dt.now()-start_run_gwas).seconds/60, 2)}')    
    
    
#    ref_dict_ls, vld_dict_ls, sst_dict_ls, ld_blk_ls, blk_size_ls = sim_ts(args)
    

