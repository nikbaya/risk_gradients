#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:52:34 2020

Runs large-scale simulations for testing PRS-CS, using independent LD blocks

To setup VM:
conda create -n msprime -y -q python=3.6.10 numpy=1.18.1 scipy=1.4.1 pandas=1.0.1 # create conda environment named msprime and install msprime dependencies
conda activate msprime # activate msprime environment
conda install -y -c conda-forge msprime=0.7.4 # install msprime Python package
wget -O msprime_prs_ldblk.py https://raw.githubusercontent.com/nikbaya/risk_gradients/master/python/msprime_prs_ldblk.py && chmod +x msprime_prs_ldblk.py

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


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_ref', default=1000, type=int,
        help='Number of individuals in reference panel.')
parser.add_argument('--n_gwas', default=10000, type=int,
        help='Number of individuals in the discovery GWAS.')
parser.add_argument('--n_train', default=10000, type=int,
        help='Number of individuals used to train the PLINK clumping.')
parser.add_argument('--n_test', default=2000, type=int,
        help='Number of individuals in the holdout set for testing PRS.')
parser.add_argument('--m_total', default=1000000, type=int,
        help='Total number of base pairs to simulate.')
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
        'of @ with chromosome numbers. To use default recombination maps, pass '
        '"1" as the argument to this flag')
parser.add_argument('--rec_rate_thresh', default=50, type=float,
        help='Recombination rate.')
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
                logfile += f'mtotal_{args.m_total}.nchr_{args.n_chr}.h2_{args.h2_A}.'
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
                       msprime.PopulationConfiguration(sample_size=N_haps[2], initial_size=N_AS, growth_rate=r_AS)]

        migration_mat = [[0, m_AF_EU, m_AF_AS],
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
                              msprime.PopulationParametersChange(time=T_AF, initial_size=N_A, population_id=0)
                                                ]
        # Return the output required for a simulation study.
        return pop_configs, migration_mat, demographic_events, N_A, n_pops

def create_ld_blocks(args, rec_map_path):
        # load recombination maps (from https://github.com/nikbaya/risk_gradients/tree/master/data)
        peak_radius = 500000 # radius in base pairs around a local peak in recombination rate
        max_ldblk_len = peak_radius*10 # maximum length of an LD block in base pairs
#        rec_map_list = [[] for _ in range(args.n_chr)] # list of lists of recmap objects for each LD block for each chromosome
        first_position_list = [[] for _ in range(args.n_chr)]# list of lists of positions for base pairs at the start of each LD block for each chromosome
        if args.rec_map:
                positions_list = []
                a_pos = 0
                rates = []
                recmap_df_list = []
                max_position_list = [] # list of maximum base pair positions
                m_total_possible = 0 # total number of base pairs available in recmaps for selected number of chromosomes
                for chr_idx in range(args.n_chr):
                        rec_map_fname = rec_map_path.replace('@',str(chr_idx+1))
                        recmap_df = pd.read_csv(rec_map_fname, delim_whitespace=True)
                        recmap_df = recmap_df.rename(columns={'COMBINED_rate(cM/Mb)':'rate'})
                        recmap_df_list.append(recmap_df)
                        max_position = recmap_df.position.max()
                        max_position_list.append(max_position)
                        m_total_possible += max_position # assumes that maximum base pair position is last entry in recmap
                for chr_idx in range(args.n_chr):
                        positions_list.append([])
                        recmap_df = recmap_df_list[chr_idx]
#                        m_chr = args.m_total*(max_position_list[chr_idx]/m_total_possible) # number of base pairs to simulate on current chromosome
#                        print(f'm_chr: {m_chr}')
                        first_position = 0 # first position of window defining current LD block (left-most side of window)
                        first_position_list[chr_idx].append(first_position)
                        peak_idx = None # index of hotspot in recmap dataframe
                        peak_position = None # position in base pairs of current peak
                        peak_rate = args.rec_rate_thresh # start at baseline of rec rate threshold
                        hotspot_idx_list = [] # list of indices of hotspots
                        recmap_positions = recmap_df.position.values # list of positions in recmap
                        recmap_rates = recmap_df.rate
                        for idx, position, rate in zip(recmap_df.index, recmap_positions, recmap_rates):
                                if peak_position == None and rate > peak_rate: # if no peak position has been found yet and current position has recombination rate > threshold
                                        peak_idx = idx    
                                        peak_position = position
                                        peak_rate = rate
                                        continue
                                if peak_position != None and (position > peak_position+peak_radius or position > first_position+max_ldblk_len): # if current position is outside of peak radius and max ld block length
                                        hotspot_idx_list.append(peak_idx)
                                        first_position = position # first position of window defining current LD block (left-most side of window)
                                        first_position_list[chr_idx].append(first_position)
                                        # reset for new LD block
                                        peak_idx = None # index of hotspot in recmap dataframe
                                        peak_position = None # position in base pairs of current peak
                                        peak_rate = args.rec_rate_thresh # start at baseline of rec rate threshold
                                elif rate > peak_rate: # update if still in ld block and at a new maximum rate
                                        peak_idx = idx    
                                        peak_position = position
                                        peak_rate = rate
                        print(len(hotspot_idx_list))
                        if hotspot_idx_list[0] != 0: 
                                hotspot_idx_list.insert(0,0) # create fake hotspot at first base pair position if it doesn't exist (useful for creating LD blocks later)
                        if hotspot_idx_list[-1] != recmap_df.index.max():
                                hotspot_idx_list.append(recmap_df.index.max()) # create fake hotspot at last base pair position if it doesn't exist (useful for creating LD blocks later)
                        for left_hotspot_idx, right_hotspot_idx in zip(hotspot_idx_list[:-1], hotspot_idx_list[1:]):
                                positions_ldblk = recmap_positions[left_hotspot_idx:right_hotspot_idx]
                                positions_ldblk -= positions_ldblk.min()
                                positions_ldblk += a_pos
                                a_pos = positions_ldblk[-1]+1
                                positions_ldblk = positions_ldblk.tolist()
                            
#                                positions_ldblk = [a_pos, recmap_positions[right_hotspot_idx]]
#                                a_pos = recmap_positions[right_hotspot_idx]+1
                                                                
                                rates_ldblk = (recmap_rates[left_hotspot_idx:right_hotspot_idx]/1e8).tolist() # convert to base pair and per-generation scale
                                rates_ldblk[-1] = 0.5 # 
#                                rates_ldblk = [args.rec, 0.5]

                                positions_list[chr_idx] += positions_ldblk
                                rates += rates_ldblk
#                                num_loci = m_chr*(max(positions)-min(positions))/max_position_list[chr_idx] # number of loci for ld block scaled by size of ld block relative to current chrom
                print(len(positions_list[0]))
                positions = [pos for pos_chr in positions_list for pos in pos_chr]
                rates[-1] = 0
                print(len(positions))
                print(len(rates))
                print(positions[130:140])
                print(rates[130:140])
                num_loci = min(args.m_total, sum(max_position_list) // 100)
        else: # if not using real recombination maps, use uniform recombination map
                positions = []
                b_pos = -1
                n_ldblks_per_chr = 1 # number of LD blocks per chromosome
#                m_chr = int(args.m_total/args.n_chr)
#                print(m_chr)
                for chr_idx in range(args.n_chr):
                        for ldblk_idx in range(n_ldblks_per_chr):
                                a_pos = b_pos+1
                                b_pos = a_pos + 3e9//args.n_chr//n_ldblks_per_chr
                                positions += [a_pos, b_pos]
                rates = [args.rec if x%2==0 else 0.5 for x in range(2*n_ldblks_per_chr*args.n_chr)]
                rates[-1] = 0
                num_loci = args.n_chr*n_ldblks_per_chr #args.m_total #positions[-1] // 100
                first_position_list = positions[::2]
                
#        print(positions)
#        print(rates)
        rec_map = msprime.RecombinationMap(positions=positions,
                                           rates=rates,
                                           num_loci=num_loci)
                
#        for chr_idx in range(args.n_chr):                
#                print(f'LD blocks in chrom {chr_idx+1}: {len(first_position_list[chr_idx])}',
#                      f'(length: mean={round(np.diff(first_position_list[chr_idx]).mean())},',
#                      f'std={round(np.diff(first_position_list[chr_idx]).std())})')

        return rec_map, first_position_list
            
def _msprime_sim(rec_map_idx):
        rec_map = rec_map_list[chr_idx][rec_map_idx]
        ts_all = msprime.simulate(sample_size=None, #set to None because sample_size info is stored in pop_configs
                                  population_configurations=pop_configs,
                                  migration_matrix=migration_mat,
                                  demographic_events=demographic_events,
                                  recombination_map=rec_map,
                                  length=None,
                                  Ne=Ne,
                                  recombination_rate=None,
                                  mutation_rate=args.mut,
                                  random_seed=random_seed)
        sim_maf = 0.01
        ts_all = get_common_mutations_ts(ts_all, maf=sim_maf, args=args) # comment out to run later phenotype simulation with causal SNPs not genotyped
        
        return ts_all

def sim_ts(m_ldblk_list, ts_all_list, m_total, chr_idx):
        r'''
        Simulate tree sequences using out-of-Africa model for a given chromosome
        '''
#        n_threads = cpu_count() # number of threads to use
#        pool = Pool(n_threads)
#        ts_all_chr_list = pool.map(_msprime_sim, range(len(rec_map_list[chr_idx]))) # parallelize
#
        ts_all_chr_list = []
        for rec_map_idx, rec_map in enumerate(rec_map_list[chr_idx]):
                ts = _msprime_sim(rec_map_idx) if rec_map is not None else None
                if ts is not None:
                        print(ts.num_trees)
                print(ts.first().draw(format='unicode'))
                ts_all_chr_list.append(ts)
        
        for ts_all in ts_all_chr_list:
                if ts_all is not None:
                        m_ldblk = int(ts_all.get_num_mutations())
                else:
                        m_ldblk = 0
                m_ldblk_list[chr_idx].append(m_ldblk)
                m_total += m_ldblk
                ts_all_list.append(ts_all if m_ldblk>0 else None) # if no mutations in LD block don't append ts_all

        return ts_all_list, m_ldblk_list, m_total
    
def split(ts_list_both, n1):
        r'''
        split `ts_list_both` into two, with the first half containing the first
        `n1` samples.
        '''
        ts_list1 = [ts.simplify(samples=ts.samples()[:2*n1]) for ts in ts_list_both] # first 2*args.n_ref samples in tree sequence are ref individuals
        ts_list2 = [ts.simplify(samples=ts.samples()[2*n1:]) for ts in ts_list_both] # all but first 2*args.n_ref samples in tree sequence are non-ref individuals

        return ts_list1, ts_list2
    
def joint_maf_filter(*ts_lists, args, maf=0.01):
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
                

if __name__=="__main__":
        args = parser.parse_args()

        to_log(args=args, string=f'start time: {dt.now().strftime("%d/%m/%Y %H:%M:%S")}')
        to_log(args=args, string=args)

        assert args.sbrprior in ['def','inf','ss'], 'ERROR: --sbrprior {args.sbrprior} is not allowed'

        # download gctb and plink
        gctb_path, plink_path, rec_map_path = get_downloads(args=args)

        # set up recombination maps for ld blocks
        to_log(args=args, string=f'\n... creating LD blocks ...')
        start_create_ldblks = dt.now()
        rec_map, first_position_list = create_ld_blocks(args, rec_map_path)
        to_log(args=args, string=f'creating LD blocks time: {round((dt.now()-start_create_ldblks).seconds/60, 2)} min\n')
        
        # simulate with out-of-Africa model
        n_total = args.n_gwas + args.n_test + args.n_ref
        sample_size = [0, n_total, 0] #only set EUR (2nd element in list) sample size to be greater than 0
        pop_configs, migration_mat, demographic_events, Ne, n_pops = _out_of_africa(sample_size)

        # simulate tree sequences 
        to_log(args=args, string=f'... starting tree sequence sim ...')
        start_sim_ts = dt.now()
        
#        print(Ne)
        sample_size = 1000 # args.n_gwas+args.n_ref+args.n_train+args.n_test
#        ts_all = msprime.simulate(sample_size=None, #set to None because sample_size info is stored in pop_configs
#                          population_configurations=pop_configs,
#                          migration_matrix=migration_mat,
#                          demographic_events=demographic_events,
#                          recombination_map=rec_map,
#                          length=None,
#                          Ne=Ne,
#                          recombination_rate=None,
#                          mutation_rate=args.mut,
#                          random_seed=args.seed)
#        print(sample_size)
        ts_all = msprime.simulate(sample_size=sample_size, 
                                  Ne=1000, 
                                  recombination_map=rec_map,
                                  mutation_rate = args.mut,
                                  model='hudson')
        sim_maf = 0.01
        ts_all = get_common_mutations_ts(ts_all, maf=sim_maf, args=args) # comment out to run later phenotype simulation with causal SNPs not genotyped
        print(f'num mutations: {int(ts_all.get_num_mutations())}')
        to_log(args=args, string=f'sim_ts time: {round((dt.now()-start_sim_ts).seconds/60, 2)} min\n')
        
        assert False
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        m_ldblk_list = [[] for _ in range(args.n_chr)] # list of lists of number of mutations per LD block per chromosome
        ts_all_list = [[] for _ in range(args.n_chr)] # list of lists of tree sequences per LD block per chromosome
        m_total = 0 # total number of SNPs post MAF filter
        for chr_idx in range(args.n_chr):
                random_seed = (args.seed+chr_idx) % 2**32 if args.seed is not None else args.seed # must be less than 2^32
    
                ts_all_list, m_ldblk_list, m_total = sim_ts(m_ldblk_list=m_ldblk_list, 
                                                            ts_all_list=ts_all_list, 
                                                            m_total=m_total, 
                                                            chr_idx=chr_idx)
                
                to_log(args=args, string=f'number of sites in chr {chr_idx+1}: {sum(m_ldblk_list[chr_idx])}')
                if sum(m_ldblk_list[chr_idx])==0:
                    print(rec_map_list[chr_idx])
                to_log(args=args, string=f'running total of sites : {m_total}')
        
#        # split into ref and non-ref
#        for chr_idx in range(args.n_chr):
#                ts_all_chr_list = ts_all_list[chr_idx]
#                for ts_all in ts_all_chr_list: # for each LD block in chromosome
#                        ts_ref_list, ts_nonref_list = split(ts_list_both, args.n_ref)
        
        # simulate phenotype
        
'''
def create_ld_blocks(args, rec_map_path):
        # load recombination maps (from https://github.com/nikbaya/risk_gradients/tree/master/data)
        peak_radius = 500000 # radius in base pairs around a local peak in recombination rate
        max_ldblk_len = peak_radius*10 # maximum length of an LD block in base pairs
        rec_map_list = [[] for _ in range(args.n_chr)] # list of lists of recmap objects for each LD block for each chromosome
        first_position_list = [[] for _ in range(args.n_chr)]# list of lists of positions for base pairs at the start of each LD block for each chromosome
        if args.rec_map:
                recmap_df_list = []
                max_position_list = [] # list of maximum base pair positions
                m_total_possible = 0 # total number of base pairs available in recmaps for selected number of chromosomes
                for chr_idx in range(args.n_chr):
                        rec_map_fname = rec_map_path.replace('@',str(chr_idx+1))
                        recmap_df = pd.read_csv(rec_map_fname, delim_whitespace=True)
                        recmap_df = recmap_df.rename(columns={'COMBINED_rate(cM/Mb)':'rate'})
                        recmap_df_list.append(recmap_df)
                        max_position = recmap_df.position.max()
                        max_position_list.append(max_position)
                        m_total_possible += max_position # assumes that maximum base pair position is last entry in recmap
                for chr_idx in range(args.n_chr):
                        recmap_df = recmap_df_list[chr_idx]
                        m_chr = args.m_total*(max_position_list[chr_idx]/m_total_possible) # number of base pairs to simulate on current chromosome
#                        print(f'm_chr: {m_chr}')
                        first_position = 0 # first position of window defining current LD block (left-most side of window)
                        first_position_list[chr_idx].append(first_position)
                        peak_idx = None # index of hotspot in recmap dataframe
                        peak_position = None # position in base pairs of current peak
                        peak_rate = args.rec_rate_thresh # start at baseline of rec rate threshold
                        hotspot_idx_list = [] # list of indices of hotspots
                        recmap_positions = recmap_df.position.values # list of positions in recmap
                        recmap_rates = recmap_df.rate.tolist() # list of rates in recmap
                        for idx, position, rate in zip(recmap_df.index, recmap_positions, recmap_rates):
                                if peak_position == None and rate > peak_rate: # if no peak position has been found yet and current position has recombination rate > threshold
                                        peak_idx = idx    
                                        peak_position = position
                                        peak_rate = rate
                                        continue
                                if peak_position != None and (position > peak_position+peak_radius or position > first_position+max_ldblk_len): # if current position is outside of peak radius and max ld block length
                                        hotspot_idx_list.append(peak_idx)
                                        first_position = position # first position of window defining current LD block (left-most side of window)
                                        first_position_list[chr_idx].append(first_position)
                                        # reset for new LD block
                                        peak_idx = None # index of hotspot in recmap dataframe
                                        peak_position = None # position in base pairs of current peak
                                        peak_rate = args.rec_rate_thresh # start at baseline of rec rate threshold
                                elif rate > peak_rate: # update if still in ld block and at a new maximum rate
                                        peak_idx = idx    
                                        peak_position = position
                                        peak_rate = rate
#                        print(hotspot_idx_list)
                        if hotspot_idx_list[0] != 0: 
                                hotspot_idx_list.insert(0,0) # create fake hotspot at first base pair position if it doesn't exist (useful for creating LD blocks later)
                        if hotspot_idx_list[-1] != recmap_df.index.max():
                                hotspot_idx_list.append(recmap_df.index.max()) # create fake hotspot at last base pair position if it doesn't exist (useful for creating LD blocks later)
                        for left_hotspot_idx, right_hotspot_idx in zip(hotspot_idx_list[:-1], hotspot_idx_list[1:]):
                                positions = recmap_positions[left_hotspot_idx:right_hotspot_idx]
#                                print(type(positions))
                                if positions[0]>1e6:
#                                        print(positions[0])
                                        break
                                positions -= positions.min()
                                rates = recmap_rates[left_hotspot_idx:right_hotspot_idx]        
                                rates[-1] = 0
                                num_loci = m_chr*(max(positions)-min(positions))/max_position_list[chr_idx] # number of loci for ld block scaled by size of ld block relative to current chrom
#                                print(num_loci)
                                num_loci = 1 # max(1, int(num_loci))
                                if num_loci > 0:
                                        rec_map = msprime.RecombinationMap(positions=positions.tolist(),
                                                                           rates=rates,
                                                                           num_loci=None)
                                else:
                                        rec_map = None
                                rec_map_list[chr_idx].append(rec_map)
        else: # if not using real recombination maps, use uniform recombination map
                for chr_idx in range(args.n_chr):
                        m_chr = int(args.m_total/args.n_chr)
                        n_ldblks_per_chr = 100 # number of LD blocks per chromosome
                        num_loci = int(m_chr/n_ldblks_per_chr) # num loci per LD block
                        for ldblk_idx in range(n_ldblks_per_chr):
                                rec_map = msprime.RecombinationMap.uniform_map(length=num_loci,
                                                                               rate=args.rec,
                                                                               num_loci=num_loci)
                                rec_map_list[chr_idx].append(rec_map)
                        first_position_list[chr_idx] = (np.arange(0,n_ldblks_per_chr)*num_loci).tolist()
        for chr_idx in range(args.n_chr):                
                print(f'LD blocks in chrom {chr_idx+1}: {len(first_position_list[chr_idx])}',
                      f'(length: mean={round(np.diff(first_position_list[chr_idx]).mean())},',
                      f'std={round(np.diff(first_position_list[chr_idx]).std())})')

        return rec_map_list, first_position_list
'''