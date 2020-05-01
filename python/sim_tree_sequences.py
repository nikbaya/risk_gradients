#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 07:39:01 2020

Runs large-scale tree sequence simulations using:
    - Flat recombination map
    - Modeling chromosomes as separate tree sequences
    - Hybrid simulations (discrete-time Wright-Fisher for recent past, coalescent for ancient past)

To setup VM:
conda create -n msprime -y -q python=3.6.10 numpy=1.18.1 scipy=1.4.1 pandas=1.0.1 # create conda environment named msprime and install msprime dependencies
conda activate msprime # activate msprime environment
conda install -y -c conda-forge msprime=0.7.4 # install msprime Python package

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
import tskit # installed with msprime, v0.2.3
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_ref', default=1000, type=int,
        help='Number of individuals in reference panel.')
parser.add_argument('--n_gwas', default=10000, type=int,
        help='Number of individuals in the discovery GWAS.')
parser.add_argument('--n_train', default=10000, type=int,
        help='Number of individuals used to train the PLINK clumping.')
parser.add_argument('--n_test', default=2000, type=int,
        help='Number of individuals in the holdout set for testing PRS.')
#parser.add_argument('--m_total', default=1000000, type=int,
#        help='Total number of base pairs to simulate.')
parser.add_argument('--n_chr', default=1, type=int,
        help='Number of chromosomes.')
parser.add_argument('--maf', default=0.05, type=float,
        help='The minor allele frequency cut-off.')
parser.add_argument('--rec', default=2e-8, type=float,
        help='Recombination rate across the region.')
parser.add_argument('--mut', default=2e-8, type=float,
        help='Mutation rate across the region.')
#parser.add_argument('--h2_A', default=0.3, type=float,
#        help='Additive heritability contribution.')
#parser.add_argument('--p_causal', default=1, type=float,
#        help='Proportion of SNPs that are causal.')
#parser.add_argument('--exact_h2', default=False, action='store_true',
#        help='Will set simulated phenotype to have almost exactly the right h2')
#parser.add_argument('--sim_after_maf', default=False, action='store_true',
#        help='Will simulate phenotype on MAF-filtered SNPs.'
#        'Otherwise the phenotype is simulated on all SNPs, which are then'
#        'MAF filtered before the GWAS is run')
parser.add_argument('--rec_map', default=False, type=str,
        help='If you want to pass a recombination map, include the filepath here. '
        'The filename should contain the symbol @, msprimesim will replace instances '
        'of @ with chromosome numbers. To use default recombination maps, pass '
        '"1" as the argument to this flag')
#parser.add_argument('--rec_rate_thresh', default=50, type=float,
#        help='Recombination rate.')
#parser.add_argument('--sbr', action='store_true', default=False,
#                    help='Whether to run SBayesR (default: False)')
#parser.add_argument('--sbrprior', default='def', type=str,
#        help='Which prior to use for SBayesR. Options: def, ss, inf (default=def)')
parser.add_argument('--seed', default=None, type=int,
        help='Seed for replicability. Must be between 1 and (2^32)-1') # random seed is changed for each chromosome when calculating true SNP effects
parser.add_argument('--verbose', '-v', action='store_true', default=False,
                    help='verbose flag')


def to_log(args, string):
        r'''
        Prints string and sends it to the log file
        '''
        if args is not None:
                logfile =  f'ngwas_{args.n_gwas}.nref_{args.n_ref}.ntrain_{args.n_train}.ntest_{args.n_test}.'
                logfile += f'nchr_{args.n_chr}.rec_{args.rec}.mut_{args.mut}.'
                logfile += f'seed_{args.seed}.log'
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
        home = str(os.getcwd())
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

#        if args.rec_map:
#                if Path(args.rec_map.replace('@','1')).exists(): # only check chr 1
#                        rec_map_path = args.rec_map
#                else:
        recmap_dir = home+'/recmaps'
        recmap_wget_url = 'https://raw.githubusercontent.com/nikbaya/risk_gradients/master/data/genetic_map_chr@_combined_b37.txt'
        for chr_idx in range(22):
                chr_recmap_wget_url = recmap_wget_url.replace("@",f"{chr_idx+1}")
                if not Path(f'{recmap_dir}/{chr_recmap_wget_url.split("/")[-1]}').exists():
                        exit_code = subprocess.call(f'wget --quiet -nc {chr_recmap_wget_url} -P {recmap_dir}'.split())
                        assert exit_code==0, f'wget when downloading recmap for chr {chr_idx+1} failed (exit code: {exit_code})'
                        print(f'downloaded recmap for chr {chr_idx+1} (b37)')
        rec_map_path = f'{recmap_dir}/{recmap_wget_url.split("/")[-1]}'
#        else:
#            rec_map_path = None

        return gctb_path, plink_path, rec_map_path

def get_common_mutations_ts(tree_sequence, maf=0.05, args=None):
#        to_log(args=args, string=f'filtering to SNPs w/ MAF>{maf}')
        # Get the mutations > MAF.
        n_haps = tree_sequence.get_sample_size()

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
    
def _out_of_africa(sample_size, no_migration=True):
        r'''
        `sample_size` is a list of length 3: [n_afr, n_eur, n_asn]
        '''
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

        population_configurations = [msprime.PopulationConfiguration(sample_size=N_haps[0], initial_size=N_AF),
                                     msprime.PopulationConfiguration(sample_size=N_haps[1], initial_size=N_EU, growth_rate=r_EU),
                                     msprime.PopulationConfiguration(sample_size=N_haps[2], initial_size=N_AS, growth_rate=r_AS)]

        migration_matrix = [[0, m_AF_EU, m_AF_AS],
                            [m_AF_EU, 0, m_EU_AS],
                            [m_AF_AS, m_EU_AS, 0]]

        
        demographic_events = [# CEU and CHB merge into B with rate changes at T_EU_AS
                              msprime.MassMigration(time=T_EU_AS, source=2, destination=1, proportion=1.0),
                              msprime.MigrationRateChange(time=T_EU_AS, rate=0),
                              msprime.MigrationRateChange(time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
                              msprime.MigrationRateChange(time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
                              msprime.PopulationParametersChange(time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
                              # Population B merges into YRI at T_B
                              msprime.MassMigration(time=T_B, source=1, destination=0, proportion=1.0),
                              # Size changes to N_A at T_AF
                              msprime.PopulationParametersChange(time=T_AF, initial_size=N_A, population_id=0)]
        
        # Implement hybrid model (we require dtwf for recent recombination but coalescent is more efficient for simulating all of the ancient past)
        model_switch_time = 100 # number of generations in the past to switch from dtwf to coalescent for ancient history
        demographic_events.append(msprime.SimulationModelChange(time=model_switch_time,
                                                                model='hudson')) # switching from dtwf for recent past to hudson for ancient past
        # Return the output required for a simulation study.
        return population_configurations, migration_matrix, demographic_events, N_A, n_pops

        
def sim_ts(args, chrom, m_chr, population_configurations, migration_matrix, demographic_events, Ne):
        random_seed = (args.seed+chrom) % 2**32 if args.seed is not None else args.seed # must be less than 2^32

        ts_all = msprime.simulate(sample_size=None, #set to None because sample_size info is stored in pop_configs
                                  population_configurations=population_configurations,
                                  migration_matrix=migration_matrix,
                                  demographic_events=demographic_events,
                                  recombination_map=None,
                                  length=m_chr,
                                  Ne=Ne,
                                  recombination_rate=args.rec,
                                  mutation_rate=args.mut,
                                  model='dtwf',
                                  random_seed=random_seed)
        
        n_total = args.n_gwas+args.n_ref+args.n_train+args.n_test
        wd = os.getcwd()
        ts_fname = f'n_{n_total}.rec_{args.rec}.mut_{args.mut}.seed_{args.seed}.m_{m_chr}.chr_{chrom}.ts'
        ts_all.dump(path=f'{wd}/{ts_fname}')
        to_log(args=args, string=f'wrote ts to: {wd}/{ts_fname}')
        
        if args.mut > 0:
                sim_maf = 0.01
                ts_all = get_common_mutations_ts(ts_all, maf=sim_maf, args=args) # comment out to run later phenotype simulation with causal SNPs not genotyped
                
                ts_maf_fname = f'n_{n_total}.rec_{args.rec}.mut_{args.mut}.seed_{args.seed}.maf_{sim_maf}.m_{m_chr}.chr_{chrom}.ts'
                ts_all.dump(path=f'{wd}/{ts_maf_fname}')
                to_log(args=args, string=f'wrote ts to: {wd}/{ts_maf_fname}')
        
if __name__=="__main__":
        args = parser.parse_args()

        to_log(args=args, string=f'start time: {dt.now().strftime("%d/%m/%Y %H:%M:%S")}')
        to_log(args=args, string=args)

#        assert args.sbrprior in ['def','inf','ss'], 'ERROR: --sbrprior {args.sbrprior} is not allowed'

        # download gctb, plink and recombination maps
        gctb_path, plink_path, rec_map_path = get_downloads(args=args)
        
        # simulate with out-of-Africa model
        n_total = args.n_gwas + args.n_ref + args.n_train + args.n_test
        sample_size = [0, n_total, 0] #only set EUR (2nd element in list) sample size to be greater than 0
        population_configurations, migration_matrix, demographic_events, Ne, n_pops = _out_of_africa(sample_size)

        # simulate tree sequences 
        to_log(args=args, string=f'... starting tree sequence sim ...')
        start_sim_ts = dt.now()

        for chrom in [22]:
            rec_map_fname = rec_map_path.replace('@',str(chrom))
            recmap_df = pd.read_csv(rec_map_fname, delim_whitespace=True)
            m_chr = recmap_df.position.max()
            
            sim_ts(args=args,
                   chrom=chrom,
                   m_chr=m_chr,
                   population_configurations=population_configurations,
                   migration_matrix=migration_matrix,
                   demographic_events=demographic_events,
                   Ne=Ne)
                    
        to_log(args=args, string=f'sim_ts time: {round((dt.now()-start_sim_ts).seconds/60, 2)} min\n')