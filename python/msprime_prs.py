#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb      1 14:50:37 2020

Runs large-scale simulations for testing PRS-CS.

To setup VM:
conda create -n msprime -y -q python=3.6.10 numpy=1.18.1 scipy=1.4.1 pandas=1.0.1 # create conda environment named msprime and install msprime dependencies
conda activate msprime # activate msprime environment
conda install -y -c conda-forge msprime=0.7.4 # install msprime Python package
wget -O msprime_prs.py https://raw.githubusercontent.com/nikbaya/risk_gradients/master/python/msprime_prs.py && chmod +x msprime_prs.py

@author: nbaya
"""

# TODO: (Optional) parallelization for loops over chromosomes

import argparse
from pathlib import Path
from datetime import datetime as dt
import numpy as np
from scipy import linalg, random, stats
import math
import msprime
import gzip
import subprocess
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

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
parser.add_argument('--sim_after_maf', default=False, action='store_true',
        help='Will simulate phenotype on MAF-filtered SNPs.'
        'Otherwise the phenotype is simulated on all SNPs, which are then'
        'MAF filtered before the GWAS is run')
parser.add_argument('--rec_map', default=False, type=str,
        help='If you want to pass a recombination map, include the filepath here. '
        'The filename should contain the symbol @, msprimesim will replace instances '
        'of @ with chromosome numbers.')
parser.add_argument('--sbr', action='store_true', default=False,
                    help='Whether to run SBayesR (default: False)')
parser.add_argument('--sbrprior', default='def', type=str,
        help='Which prior to use for SBayesR. Options: def, ss, inf (default=def)')
parser.add_argument('--seed', default=None, type=int,
        help='Seed for replicability. Must be between 1 and (2^32)-1') # random seed is changed for each chromosome when calculating true SNP effects
parser.add_argument('--verbose', '-v', action='store_true', default=False,
                    help='verbose flag')

def to_log(args, string):
        r'''
        Prints string and sends it to the log file
        '''
        if args is not None:
                use_recmap = True if args.rec_map else False
                logfile =  f'ngwas_{args.n_gwas}.ntest_{args.n_test}.nref_{args.n_ref}.'
                logfile += f'mperchr_{args.m_per_chr}.nchr_{args.n_chr}.h2_{args.h2_A}.'
                logfile += f'pcausal_{args.p_causal}.simaftermaf_{args.sim_after_maf}.'
                logfile += f'recmap_{use_recmap}.seed_{args.seed}.log'
                if args.verbose:
                        print(string)
        else:
                logfile = None
        if type(string) is not str:
                string = str(string)
        if logfile is not None:
                with open(logfile, 'a') as log:
                        log.write(string+'\n')

def get_downloads(args):
        r'''
        Download PLINK and GCTB
        Download rec-maps if args.rec_map is not None
        '''
        home = str(Path.home())
        software_dir = home+'/software'

        # download gctb
        gctb_path = f'{software_dir}/gctb_2.0_Linux/gctb'
        if not Path(gctb_path).exists():
                print(f'downloading gctb to {gctb_path}')
                gctb_wget_url = 'https://cnsgenomics.com/software/gctb/download/gctb_2.0_Linux.zip'
                exit_code = subprocess.call(f'wget --quiet -nc {gctb_wget_url} -P {software_dir}'.split())
                assert exit_code==0, f'wget when downloading GCTB failed (exit code: {exit_code})'
                exit_code = subprocess.call(f'unzip -q {software_dir}/gctb_2.0_Linux.zip -d {software_dir}'.split())
                assert exit_code==0, f'unzip when downloading GCTB failed (exit code: {exit_code})'
                
        # download plink
        plink_path = f'{software_dir }/plink'
        if not Path(plink_path).exists():
                print(f'downloading plink to {plink_path}')
                plink_wget_url = 'http://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20200219.zip'
                exit_code = subprocess.call(f'wget --quiet -nc {plink_wget_url} -P {software_dir}'.split())
                assert exit_code==0, f'wget when downloading PLINK failed (exit code: {exit_code})'
                exit_code = subprocess.call(f'unzip -q {software_dir}/plink_linux_x86_64_20200219.zip -d {software_dir}'.split())
                assert exit_code==0, f'unzip when downloading PLINK failed (exit code: {exit_code})'
                
        if args.rec_map:
                if Path(args.rec_map.replace('@','1')).exists(): # only check chr 1
                        rec_map_path = args.rec_map
                else:
                        recmap_dir = home+'/recmaps'
                        recmap_wget_url = 'https://raw.githubusercontent.com/nikbaya/risk_gradients/master/data/genetic_map_chr@_combined_b37.txt'
                        for chr_idx in range(args.n_chr):
                                chr_recmap_wget_url = recmap_wget_url.replace("@",f"{chr_idx+1}")
                                if not Path(f'{recmap_dir}/{chr_recmap_wget_url.split("/")[-1]}').exists():
                                        exit_code = subprocess.call(f'wget --quiet -nc {chr_recmap_wget_url} -P {recmap_dir}'.split())
                                        assert exit_code==0, f'wget when downloading recmap for chr {chr_idx+1} failed (exit code: {exit_code})'
                                        print(f'downloaded recmap for chr {chr_idx+1} (b37)')
                        rec_map_path = f'{recmap_dir}/{recmap_wget_url.split("/")[-1]}'
        else:
            rec_map_path = None

        return gctb_path, plink_path, rec_map_path

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

def sim_ts(args, rec_map_path):
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

        # load recombination maps (from https://github.com/nikbaya/risk_gradients/tree/master/data)
        if args.rec_map:
                rec_map_list = []
                for chr_idx in range(args.n_chr):
                        rec_map_fname = rec_map_path.replace('@',str(chr_idx+1))
                        # TODO: Truncate genetic map to only use m base pairs
                        positions = []
                        rates = []
                        with open(rec_map_fname, 'r') as rec_map_file:
                                next(rec_map_file)  #skip header
                                for i, line in enumerate(rec_map_file):
                                        vals = line.split()
                                        if float(vals[0]) >= args.m_per_chr: # if base-pair position greater than m_per_chr
                                                break
                                        else:
                                                positions += [float(vals[0])]
                                                rates += [float(vals[1])/1e8] # convert to base pair scale and per-generation scale
                        if len(positions)>1 and len(rates)>1:
                                    rates[-1] = 0
                                    if positions[0] != 0:
                                                positions.insert(0,0)
                                                rates.insert(0,0)
                                    else:
                                                rates[0] = 0
                                    rec_map = msprime.RecombinationMap(positions=positions,
                                                                       rates=rates,
                                                                       num_loci=args.m_per_chr)
                        else:
                                    rec_map = msprime.RecombinationMap.uniform_map(length=args.m_per_chr,
                                                                                   rate=args.rec,
                                                                                   num_loci=args.m_per_chr)
                        rec_map_list.append(rec_map)
                args.rec = None
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
                random_seed = (args.seed+chr_idx) % 2**32 if args.seed is not None else args.seed # must be less than 2^32
                ts_list_all.append(msprime.simulate(sample_size=None, #set to None because sample_size info is stored in pop_configs
                                                    population_configurations=pop_configs,
                                                    migration_matrix=migration_mat,
                                                    demographic_events=demographic_events,
                                                    recombination_map=rec_map_list[chr_idx],
                                                    length=None if args.rec_map else args.m_per_chr, 
                                                    Ne=Ne,
                                                    recombination_rate=args.rec,
                                                    mutation_rate=args.mut,
                                                    random_seed=random_seed))

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
        to_log(args=args, string=f'p-causal is {args.p_causal}')

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
                # to_log(args=args, string=f'picking causal variants and determining effect sizes in chromosome {chr_idx+1}')
                # to_log(args=args, string=f'p-causal is {args.p_causal}')
                ts_pheno_A, m_causal_A, causal_A_idx = set_mutations_in_tree(ts, args.p_causal)
                m_chr = int(ts.get_num_mutations())
                to_log(args=args, string=f'chr {chr_idx+1} causal: {m_causal_A}/{m_chr}')
                beta_A = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_total * args.p_causal)), size=m_causal_A)
                beta_A_list.append(beta_A)
                ts_pheno_A_list.append(ts_pheno_A)
                causal_A_idx_list.append(causal_A_idx)

                # additive model for phenotype
                for k, variant in enumerate(ts_pheno_A.variants()): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
                                X_A = nextSNP_add(variant)
                                y += X_A * beta_A[k]

        to_log(args=args, string=f'\tm_total: {m_total}')
        m_causal_total = sum([sum(causal_A_idx) for causal_A_idx in causal_A_idx_list])
        to_log(args=args, string=f'\tm_causal_total: {m_causal_total}')

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
        pval_A_list = [None for i in range(args.n_chr)] # list of np arrays (one for each chromosome) holding GWAS p-values
        se_A_list = [None for i in range(args.n_chr)] # list of np arrays (one for each chromosome) holding GWAS standard errors
        maf_A_list = [None for i in range(args.n_chr)] # list of np arrays (one for each chromosome) holding GWAS MAF
#        pval_A_list = [None]*args.n_chr # list of np arrays (one for each chromosome) holding GWAS pvals

        n_gwas = int(ts_list_gwas[0].get_sample_size()/2 ) # assume that sample size is same across chromosomes
        for chr_idx in range(args.n_chr):
                m_gwas = ts_list_gwas[chr_idx].get_num_mutations()
                betahat_A = np.empty(shape=m_gwas)
                pval_A = np.empty(shape=m_gwas)
                se_A = np.empty(shape=m_gwas)
                maf_A = np.empty(shape=m_gwas)
                to_log(args=args, string=f'Determining beta-hats in chromosome {chr_idx+1}')
                for k, variant in enumerate(ts_list_gwas[chr_idx].variants()):
                                X_A = nextSNP_add(variant, index=None)
                                betahat, _, _, pval, stderr = stats.linregress(x=X_A, y=y.reshape(n_gwas,))
                                betahat_A[k] = betahat
                                pval_A[k] = pval
                                se_A[k] = stderr
                                af = variant.genotypes.astype(int).mean()
                                maf = min(af, 1-af)
                                maf_A[k] = maf
                betahat_A_list[chr_idx] = betahat_A
                pval_A_list[chr_idx] = pval_A
                se_A_list[chr_idx] = se_A
#                pval_A_list[chr_idx] = pval_A
                maf_A_list[chr_idx] = maf_A

        return betahat_A_list, maf_A_list, pval_A_list, se_A_list

def write_betahats(args, ts_list, beta_list, pval_list, se_list, betahat_fname):
        r'''
        Write beta-hats to file in the .ma format: https://cnsgenomics.com/software/gctb/#SummaryBayesianAlphabet
        '''
        with open(betahat_fname,'w') as betahat_file:
            betahat_file.write('SNP A1 A2 freq b se p N\n')
            n_haps = ts_list[0].get_sample_size()
            for chr_idx in range(args.n_chr):
                    betahat = beta_list[chr_idx]
                    pval = pval_list[chr_idx]
                    se = se_list[chr_idx]
                    snp_idx = 0
                    assert len(betahat)==len([v for v in ts_list[chr_idx].variants()])
                    for k, variant in enumerate(ts_list[chr_idx].variants()): 
                            gt = (np.array(variant.genotypes[0::2].astype(int)) + np.array(variant.genotypes[1::2].astype(int)))
                            af = np.mean(gt)/2 # frequency of non-ancestral allele (note: non-ancestral allele is effect allele, denoted as '1' in the A1 field)
                            betahat_file.write(f'{chr_idx+1}:{variant.site.position} {1} {0} {af} {betahat[snp_idx]} {se[snp_idx]} {pval[snp_idx]} {int(n_haps/2)}\n')
                            snp_idx += 1

def _from_vcf(betahat, plink_path, chr_idx):
        r'''
        For parallelized exporting from VCF and updating bim file with SNP IDs
        '''
        chr_betahat = betahat[betahat.CHR==(chr_idx+1)].reset_index(drop=True)
        vcf_fname = f"{bfile}.chr{chr_idx+1}.vcf.gz"
        with gzip.open(vcf_fname , "wt") as vcf_file:
                ts_list_ref[chr_idx].write_vcf(vcf_file, ploidy=2, contig_id=f'{chr_idx+1}')
        if args.n_chr > 1:
                chr_bfile_fname = f'{bfile}.chr{chr_idx+1}'
        elif args.n_chr==1:
                chr_bfile_fname = bfile
        subprocess.call(f'{plink_path} --vcf {vcf_fname } --double-id --silent --make-bed --out {chr_bfile_fname}'.split())
        chr_bim = pd.read_csv(f'{chr_bfile_fname}.bim', delim_whitespace=True,
                              names=['CHR','SNP_old','CM','BP','A1','A2']) # name SNP ID field "SNP_old" to avoid name collision when merging with betahats
        assert chr_bim.shape[0]==chr_betahat.shape[0], f'chr_bim rows: {chr_bim.shape[0]} chr_betahats rows: {chr_betahat.shape[0]}' # check that file lengths are the same
        merged = chr_bim.join(chr_betahat[['SNP']], how='inner') # only take SNP ID field from betahats
        merged = merged[['CHR','SNP','CM','BP','A1','A2']]
        merged.to_csv(f'{chr_bfile_fname}.bim',sep='\t',index=False,header=False) # overwrite existing bim file

def write_to_plink(args, ts_list, bfile, betahat_fname, plink_path):
        r'''
        Write ts_list files to bfile and merge across chromosomes if necessary
        '''
        betahat = pd.read_csv(betahat_fname, delim_whitespace=True)
        betahat['CHR'] = betahat.SNP.str.split(':',expand=True)[0].astype(int)
        _from_vcf_map = partial(_from_vcf, betahat, plink_path) # allows passing multiple arguments to from_vcf when parallelized
        n_threads = cpu_count() # can set to be lower if needed
        pool = Pool(n_threads)
        # TODO: Find a way to pass all args to limit scope of _from_vcf
        chrom_idxs = range(args.n_chr) # list of chromosome indexes (0-indexed)
        pool.map(_from_vcf_map, chrom_idxs) # parallelize    
        pool.close()
        pool.join()
        
        if args.n_chr>1:
                mergelist_fname=f'{bfile}.mergelist.txt'
                with open(mergelist_fname,'w') as mergelist_file:
                        mergelist_file.write('\n'.join([f'{bfile}.chr{chr_idx+1}' for chr_idx in range(args.n_chr)]))
                subprocess.call(f'{plink_path} --silent --merge-list {mergelist_fname} --make-bed --out {bfile}'.split())
                
def plink_clump(args, ts_list, bfile, betahat_fname, plink_path, betahat_list):
        r'''
        Run PLINK --clump to get set of SNPs for PRS
        '''
        clump_p1 = 1
        clump_p2 = 1
        clump_r2 = 0.1
        clump_kb = 500
        exit_code = subprocess.call( # get exit code
        f'''{plink_path} \
        --silent \
        --bfile {bfile} \
        --clump {betahat_fname} \
        --clump-field p \
        --clump-p1 {clump_p1} \
        --clump-p2 {clump_p2} \
        --clump-r2 {clump_r2} \
        --clump-kb {clump_kb} \
        --out {bfile}'''.split())
        
        assert exit_code==0, f'PLINK clumping failed, exit code: {exit_code}'
        
        to_log(args=args, string='converting PLINK --clump results into beta-hats')
        clumped_betahat_list = [[0 for tree in ts_list_ref[chr_idx].trees() for site in tree.sites()] for chr_idx in range(args.n_chr)] # list of lists of beta-hats for SNPs on each chromosome, initialize all as zero
        clumped = pd.read_csv(f'{bfile}.clumped', delim_whitespace=True) # initially sorted by SNP significance
        clumped = clumped.sort_values(by='BP') # sort by base pair position
        for chr_idx in range(args.n_chr):
                chr_positions = [site.position for tree in ts_list_test[chr_idx].trees() for site in tree.sites()] # list of SNP positions in tree sequence
                chr_betahats = betahat_list[chr_idx]
                chrom = chr_idx+1
                clumped_chr = clumped[clumped['CHR']==chrom]
                clumped_pos_list = clumped_chr['SNP'].str.split(':', expand=True)[1].astype('float').to_list() # split SNP IDs by ':', take the second half and keep position floats as list
                clumped_betahat_list[chr_idx] = [(snp_betahat if chr_pos in clumped_pos_list else 0) for chr_pos,snp_betahat in zip(chr_positions, chr_betahats)]
        return clumped_betahat_list

def run_SBayesR(args, gctb_path, bfile, ldm_type='full'):
        r'''
        Run SBayesR on `bfile` and convert beta-hats
        '''
        assert ldm_type in {'sparse','full'}, f'ldm_type={ldm_type} not valid'
        
        to_log(args=args, string=f'calculating gctb ld matrix')
        start_gctb_ldm = dt.now()
        exit_code = subprocess.call(
        f'''{gctb_path} \
        --bfile {bfile} \
        --make-{ldm_type}-ldm \
        --out {bfile}'''.split(),
        )
        to_log(args=args, string=f'make gctb ldm time: {round((dt.now()-start_gctb_ldm).seconds/60, 2)} min\n')
        
        assert exit_code==0, f'make-{ldm_type}-ldm failed (exit code: {exit_code})'
        
        # NOTE: --pi values must add up to 1 and must match the number of values passed to gamma
        # NOTE: can cheat by starting hsq (heritability) with true heritability by adding the following line
        # --hsq {args.h2_A} \
        # NOTE: can cheat by starting pi (heritability) with true pi for spike and slab prior:
        # --pi {args.p_causal} \
        to_log(args=args, string=f'starting sbayesr')
        start_sbayesr = dt.now()
        if args.sbrprior == 'def':
            # Default 
            cmd = f'''{gctb_path} \
            --sbayes R --ldm {bfile}.ldm.{ldm_type} \
            --pi 0.95,0.02,0.02,0.01 --gamma 0.0,0.01,0.1,1 \
            --gwas-summary {betahat_fname} --chain-length 10000 \
            --burn-in 2000  --out-freq 10 --out {bfile}'''
        elif args.sbrprior == 'inf':
            # Infinitesimal prior
            cmd = f'''{gctb_path} \
            --sbayes C --ldm {bfile}.ldm.{ldm_type} \
            --pi 1  \
            --gwas-summary {betahat_fname} --chain-length 10000 \
            --burn-in 2000  --out-freq 10 --out {bfile}'''
        elif args.sbrprior == 'ss':
            # Spike & slab prior
            cmd = f'''{gctb_path} \
            --sbayes C --ldm {bfile}.ldm.{ldm_type} \
            --pi {args.p_causal} \
            --gwas-summary {betahat_fname} --chain-length 10000 \
            --burn-in 2000  --out-freq 10 --out {bfile}'''
        print(cmd)
        exit_code = subprocess.call(cmd.split())        
        
        to_log(args=args, string=f'run sbayesr time: {round((dt.now()-start_sbayesr).seconds/60, 2)} min\n')
        
        assert exit_code==0, f'SBayesR failed (exit code: {exit_code})' # NOTE: this might not actually be effective

        to_log(args=args, string='converting SBayesR beta-hats')
        sbayesr_betahat_list = [[0 for tree in ts_list_ref[chr_idx].trees() for site in tree.sites()] for chr_idx in range(args.n_chr)] # list of lists of beta-hats for SNPs on each chromosome, initialize all as zero
        with open(f'{bfile}.snpRes','r') as snpRes_file:
                chr_idx = 0
                snp_idx = 0
                chr_positions = [site.position for tree in ts_list_test[chr_idx].trees() for site in tree.sites()]
                for i, line in enumerate(snpRes_file): # skip first line (header)
                        if i==0:
                                continue
                        vals = line.split()
                        # NOTE: This assumes the snpRes file is sorted by chr and bp position
                        while int(vals[2]) > chr_idx+1 and chr_idx<args.n_chr: # chrom is 3rd column in snpRes file
                                chr_idx = int(vals[2])-1
                                snp_idx = 0
                                chr_positions = [site.position for tree in ts_list_test[chr_idx].trees() for site in tree.sites()]
                        while snp_idx < len(chr_positions) and chr_positions[snp_idx] != float(vals[1].split(':')[1]):
                                snp_idx += 1
                        if chr_positions[snp_idx] == float(vals[1].split(':')[1]):
                                sbayesr_betahat_list[chr_idx][snp_idx] = float(vals[10])
                                snp_idx += 1

        return sbayesr_betahat_list


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
                        to_log(args=args, string=f'correlation between betas (chr {chr_idx+1}) : {round(r, 5)}')

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
            to_log(args=args, string=f'h2 obs. (y w/ y_gen R^2): {round(r**2, 5)}')
        else:
            to_log(args=args, string=f'y w/ yhat r^2: {round(r**2, 5)}'+(' (WARNING: r<0)' if r<0 else ''))

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

def _gigrnd(p, a, b):
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

def _mcmc(args, sst_dict_list, ld_list, chrom_idx):
        sst_dict = sst_dict_list[chrom_idx]
        ld_blk = ld_list[chrom_idx]
        blk_size = [len(blk) for blk in ld_blk]
        
        a = 1
        b = 0.5
        phi = None
        n = args.n_gwas
        n_iter = 1000
        n_burnin = 500
        thin = 5
        beta_std = True
        seed = args.seed
        
        chrom=chrom_idx+1
        
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
                        psi[jj] = _gigrnd(a-0.5, 2.0*delta[jj], n*beta[jj]**2/sigma)
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

def prs_cs(args, betahat_A_list, maf_A_list, ld_list):
        r'''
        Use PRS-CS to calculate adjusted beta-hats
        '''

        sst_dict_list = [{'BETA':betahat_A_list[chr_idx], 'MAF':maf_A_list[chr_idx]}
                                         for chr_idx in range(args.n_chr)]

#        beta_est_list = []
#        for chr_idx in range(args.n_chr):
#                sst_dict = sst_dict_list[chr_idx]
#                ld_blk = ld_list[chr_idx]
#                blk_size = [len(blk) for blk in ld_blk]
#                beta_est = mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size,
#                                  n_iter, n_burnin, thin, chr_idx+1,
#                                  beta_std, seed, args=args)
#                beta_est_list.append(beta_est)
        
        mcmc_map = partial(_mcmc, args, sst_dict_list, ld_list)

        n_threads = cpu_count() # can set to be lower if needed
        pool = Pool(n_threads)
        chrom_idxs = range(args.n_chr) # list of chromosome indexes (0-indexed)
        beta_est_list = pool.map(mcmc_map, chrom_idxs) # parallelize    
        pool.close()
        pool.join()
        
        return beta_est_list

if __name__ == '__main__':
        args = parser.parse_args()

        to_log(args=args, string=f'start time: {dt.now().strftime("%d/%m/%Y %H:%M:%S")}')
        to_log(args=args, string=args)
        
        assert args.sbrprior in ['def','inf','ss'], 'ERROR: --sbrprior {args.sbrprior} is not allowed'

        # TODO: Consider adding argument for proportion of genome that is genotyped

        # download gctb and plink
        gctb_path, plink_path, rec_map_path = get_downloads(args=args)

        # simulate tree sequences
        to_log(args=args, string=f'starting tree sequence sim')
        start_sim_ts = dt.now()
        args, ts_list_all, ts_list_geno_all, m, m_start, m_total, m_geno, m_geno_start, \
        m_geno_total, n_pops, genotyped_list_index = sim_ts(args=args, rec_map_path=rec_map_path)
        to_log(args=args, string=f'sim_ts time: {round((dt.now()-start_sim_ts).seconds/60, 2)} min\n')

        # split into ref and non-ref (non-ref will have both the gwas and test sets)
        ts_list_ref, ts_list_nonref = split(ts_list_both=ts_list_all,
                                            n1=args.n_ref)

        ## MAF filter before simulating phenotype
        if args.sim_after_maf:
                # joint MAF filter ref and non-ref
                ts_list_ref, ts_list_nonref = joint_maf_filter(ts_list_ref,
                                                               ts_list_nonref,
                                                               args=args,
                                                               maf=args.maf)

                # simulate phenotype
                to_log(args=args, string=f'starting phenotype sim')
                start_sim_phen = dt.now()
                y, beta_A_list, ts_pheno_A_list, causal_A_idx_list = sim_phen(args=args,
                                                                              n_pops=n_pops,
                                                                              ts_list=ts_list_nonref,
                                                                              m_total=m_total)
                assert y.shape[0] == args.n_gwas+args.n_test
                y_gwas = y[:args.n_gwas] # take first n_gwas individuals, just like in the splitting of ts_list_nonref
                y_test = y[args.n_gwas:] # take the complement of the first n_gwas individuals, just like in the splitting of ts_list_nonref
                # TODO: Check that individuals are in the same order in ts_pheno_A_list and ts_list_nonref
                to_log(args=args, string=f'sim_phen time: {round((dt.now()-start_sim_phen).seconds/60, 2)} min\n')

                # split non-ref into gwas and test sets
                ts_list_gwas, ts_list_test = split(ts_list_both=ts_list_nonref,
                                                   n1=args.n_gwas)

                # joint MAF filter ref, gwas, and test cohorts
                # TODO: remove necessity of passing args to this function to get n_chr
                start_joint_maf = dt.now()
                ts_list_ref, ts_list_gwas, ts_list_test = joint_maf_filter(ts_list_ref,
                                                                           ts_list_gwas,
                                                                           ts_list_test,
                                                                           args=args,
                                                                           maf=args.maf)

        ## MAF filter after simulating phenotype (reduces PRS accuracy)
        else:
                # simulate phenotype
                to_log(args=args, string=f'starting phenotype sim')
                start_sim_phen = dt.now()
                y, beta_A_list, ts_pheno_A_list, causal_A_idx_list = sim_phen(args=args,
                                                                              n_pops=n_pops,
                                                                              ts_list=ts_list_nonref,
                                                                              m_total=m_total)
                assert y.shape[0] == args.n_gwas+args.n_test
                y_gwas = y[:args.n_gwas] # take first n_gwas individuals, just like in the splitting of ts_list_nonref
                y_test = y[args.n_gwas:] # take the complement of the first n_gwas individuals, just like in the splitting of ts_list_nonref
                # TODO: Check that individuals are in the same order in ts_pheno_A_list and ts_list_nonref
                to_log(args=args, string=f'sim_phen time: {round((dt.now()-start_sim_phen).seconds/60, 2)} min\n')

                # split non-ref into gwas and test sets
                ts_list_gwas, ts_list_test = split(ts_list_both=ts_list_nonref,
                                                   n1=args.n_gwas)

                # joint MAF filter ref, gwas, and test cohorts
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
        to_log(args=args, string=f'starting gwas')
        start_run_gwas = dt.now()
        betahat_A_list, maf_A_list, pval_A_list, se_A_list = run_gwas(args=args,
                                                                      y=y_gwas,
                                                                      ts_list_gwas=ts_list_gwas)
        to_log(args=args, string=f'run gwas time: {round((dt.now()-start_run_gwas).seconds/60, 2)} min\n')

        # write beta-hats to file
        # .ma file format (required by SBayesR): SNP A1 A2 freq b se p N
        use_recmap = True if args.rec_map else False
        bfile =  f'tmp_ng{args.n_gwas}.nt{args.n_test}.nr{args.n_ref}.' # bfile prefix of PLINK files of reference set; also used as uniq identifier for simulation
        bfile += f'mpc{args.m_per_chr}.nc{args.n_chr}.h2{args.h2_A}.'
        bfile += f'p{args.p_causal}.sam{args.sim_after_maf}.'
        bfile += f'rm{use_recmap}.s{args.seed}'
        subprocess.call(f'rm {bfile}*'.split(), stderr=subprocess.DEVNULL) # remove existing files with this prefix
        betahat_fname = f'{bfile}.betahat.ma'
        write_betahats(args=args,
                       ts_list=ts_list_gwas,
                       beta_list=betahat_A_list,
                       pval_list=pval_A_list,
                       se_list=se_A_list,
                       betahat_fname=betahat_fname)

        # For adding suffix to duplicates: https://groups.google.com/forum/#!topic/comp.lang.python/VyzA4ksBj24

        # write ref samples to PLINK
        start_write_ref = dt.now()
        to_log(args=args, string=f'writing ref samples to PLINK')
        write_to_plink(args=args, ts_list=ts_list_ref, bfile=bfile,
                       betahat_fname=betahat_fname, plink_path=plink_path)
        to_log(args=args, string=f'write_to_plink time: {round((dt.now()-start_write_ref).seconds/60, 2)} min\n')
        
        # run PLINK clumping and get clumped betahats
        to_log(args=args, string=f'running PLINK clumping')
        start_plink_clump = dt.now()
        clumped_betahat_list = plink_clump(args=args, ts_list=ts_list_ref, bfile=bfile,
                                              betahat_fname=betahat_fname, plink_path=plink_path,
                                              betahat_list = betahat_A_list)
        to_log(args=args, string=f'plink_clump time: {round((dt.now()-start_plink_clump).seconds/60, 2)} min\n')

        # run SBayesR with GCTB and convert betas
        if args.sbr:
            try:
                sbayesr_betahat_list = run_SBayesR(args=args, gctb_path=gctb_path, bfile=bfile)
                sbr_successful = True
            except:
                print('SBayesR failed')                
                sbr_successful = False

        # calculate LD matrix for PRS-CS
        to_log(args=args, string=f'calculating tskit ld matrix')
        start_calc_ld = dt.now()
        ld_list = calc_ld(args=args,
                          ts_list_ref=ts_list_ref)
        to_log(args=args, string=f'calc ld time: {round((dt.now()-start_calc_ld).seconds/60, 2)} min\n')

        # run PRS-CS
        # TODO: Figure out ZeroDivisionError at s = min(1/lam, math.log(1+1/alpha+math.sqrt(1/math.pow(alpha,2)+2/alpha)))
        start_prs_cs = dt.now()
        try:
            prscs_betahat_list = prs_cs(args=args,
                                   betahat_A_list=betahat_A_list,
                                   maf_A_list=maf_A_list,
                                   ld_list=ld_list)
        except ZeroDivisionError:
            print('\nPRS-CS failed due to ZeroDivisionError\n')
            
        to_log(args=args, string=f'prs-cs time: {round((dt.now()-start_prs_cs).seconds/60, 2)} min\n')

        # calculate beta/betahat and y/yhat correlations for unadjusted GWAS
        to_log(args=args, string=f'\ncorr for unadj. betas (m={m_total})')
        calc_corr(args=args,
                  causal_idx_pheno_list=causal_idx_pheno_list,
                  causal_idx_list=causal_idx_test_list,
                  beta_est_list=betahat_A_list,
                  y_test=y_test,
                  ts_list_test=ts_list_test)
        
        # calculate beta/betahat and y/yhat correlations for clumped GWAS
        to_log(args=args, string=f'\ncorr for clumped betas (m={m_total})')
        calc_corr(args=args,
                  causal_idx_pheno_list=causal_idx_pheno_list,
                  causal_idx_list=causal_idx_test_list,
                  beta_est_list=clumped_betahat_list,
                  y_test=y_test,
                  ts_list_test=ts_list_test)


        # calculate beta/betahat and y/yhat correlations for SBayesR
        if args.sbr and sbr_successful:
            to_log(args=args, string=f'\ncorr for SBayesR (m={m_total})')
            calc_corr(args=args,
                      causal_idx_pheno_list=causal_idx_pheno_list,
                      causal_idx_list=causal_idx_test_list,
                      beta_est_list=sbayesr_betahat_list,
                      y_test=y_test,
                      ts_list_test=ts_list_test)

        # calculate beta/betahat and y/yhat correlations for PRS-CS
        to_log(args=args, string=f'\ncorr for PRS-CS (m={m_total})')
        calc_corr(args=args,
                  causal_idx_pheno_list=causal_idx_pheno_list,
                  causal_idx_list=causal_idx_test_list,
                  beta_est_list=prscs_betahat_list,
                  y_test=y_test,
                  ts_list_test=ts_list_test)

        to_log(args=args, string=f'total time: {round((dt.now()-start_sim_ts).seconds/60, 2)} min\n')

        to_log(args=args, string=args)
