#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb	1 14:50:37 2020

Runs large-scale simulations for testing PRS-CS.

To setup VM:
	conda create -n msprime -y -q python=3.6 numpy=1.18 # create conda environment named msprime and install msprime dependencies
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

def get_common_mutations_ts(tree_sequence, maf=0.05):
#		 common_sites = msprime.SiteTable()
#		 common_mutations = msprime.MutationTable()

	# Get the mutations > MAF.
	n_haps = tree_sequence.get_sample_size()
	print(f'Determining sites > MAF cutoff {maf}')

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
	n_total = args.n + args.n_ref
	sample_size = [0, n_total, 0] #only set EUR sample size to be greater than 0
	pop_configs, migration_mat, demographic_events, Ne, n_pops = out_of_africa(sample_size)

	dp = msprime.DemographyDebugger(Ne=Ne,
									population_configurations=pop_configs,
									migration_matrix=migration_mat,
									demographic_events=demographic_events)
	dp.print_history()

	for chr_idx in range(args.n_chr):
		ts_list_all.append(msprime.simulate(sample_size=None, #set to None because sample_size info is stored in pop_configs
											population_configurations=pop_configs,
											migration_matrix=migration_mat,
											demographic_events=demographic_events,
											recombination_map=rec_map_list[chr_idx],
											length=args.m_per_chr, Ne=Ne,
											recombination_rate=args.rec,
											mutation_rate=args.mut))

		# get mutations > MAF
#		 ts_list_all[chr] = get_common_mutations_ts(args, ts_list_all[chr]) # comment out to run later phenotype simulation with causal SNPs not genotyped

		m[chr_idx] = int(ts_list_all[chr_idx].get_num_mutations())
		m_start[chr_idx] = m_total
		m_total += m[chr_idx]
		print(f'Number of mutations in chr {chr_idx+1}: {m[chr_idx]}')
		print(f'Running total of sites : {m_total}')

		ts_list_geno_all.append(ts_list_all[chr_idx])
		genotyped_list_index.append(np.ones(ts_list_all[chr_idx].num_mutations, dtype=bool))
		m_geno[chr_idx] = m[chr_idx]
		m_geno_start[chr_idx] = m_start[chr_idx]
		m_geno_total = m_total
#		 print(f'Number of sites genotyped in the chr {chr+1}: {m_geno[chr]}')
#		 print(f'Running total of sites genotyped: {m_geno_total}')


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

def split(args, ts_list_all, ts_list_geno_all):
	r'''
	split into ref and non-ref subsets of the data
	'''
	ts_list_ref = [ts.simplify(samples=ts.samples()[:2*args.n_ref]) for ts in ts_list_all] # first 2*args.n_ref samples in tree sequence are ref individuals
	ts_list = [ts.simplify(samples=ts.samples()[2*args.n_ref:]) for ts in ts_list_all] # all but first 2*args.n_ref samples in tree sequence are non-ref individuals

	ts_list_geno, genotyped_list_index, m_total, m_geno_total = _update_vars(args, ts_list) # probably not necessary

	return ts_list_ref, ts_list, ts_list_geno, m, m_start, m_total, m_geno, \
		   m_geno_start, m_geno_total, genotyped_list_index

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
#		 p = mean_X / 2
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

	print(f'Additive h2 is {args.h2_A}')

	n = int(ts_list[0].get_sample_size()/2 )
	y = np.zeros(n)
	beta_A_list = [] # list of np arrays (one for each chromosome) containing true effect sizes
	ts_pheno_A_list = [] # list of tree sequences on which phenotypes are calculated (possibly ignoring invariant SNPs?)
	causal_A_idx_list = [] # list of booleans indicating if a SNP is causal

	for chr_idx in range(args.n_chr):
		m_chr = int(ts_list[chr_idx].get_num_mutations())
		print(f'Picking causal variants and determining effect sizes in chromosome {chr_idx+1}')
		print(f'p-causal is {args.p_causal}')
		ts_pheno_A, m_causal_A, causal_A_idx = set_mutations_in_tree(ts_list[chr_idx], args.p_causal)
		print(f'Picked {m_causal_A} additive causal variants out of {m_chr}')
		beta_A = np.random.normal(loc=0, scale=np.sqrt(args.h2_A / (m_chr * args.p_causal)), size=m_causal_A)
		beta_A_list.append(beta_A)
		ts_pheno_A_list.append(ts_pheno_A)
		causal_A_idx_list.append(causal_A_idx)

		# additive model for phenotype
		for k, variant in enumerate(ts_pheno_A.variants()): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
				X_A = nextSNP_add(variant)
				y += X_A * beta_A[k]

	# add noise to phenotypes
	y += np.random.normal(loc=0, scale=np.sqrt(1-(args.h2_A)), size=n)

	return y, beta_A_list, ts_pheno_A_list, causal_A_idx_list

def joint_maf_filter(ts_list1, ts_list2, maf=0.05):
		r'''
		Filter to SNPs with MAF>`maf` in both `ts_list1` and `ts_list2`
		'''
		for chr_idx in range(args.n_chr):
			ts1 = ts_list1[chr_idx]
			ts2 = ts_list2[chr_idx]

			n_haps1 = ts1.get_sample_size()
			n_haps2 = ts2.get_sample_size()
			print(f'Determining sites > MAF cutoff {args.maf}')

			tables1 = ts1.dump_tables()
			tables1.mutations.clear()
			tables1.sites.clear()

			tables2 = ts2.dump_tables()
			tables2.mutations.clear()
			tables2.sites.clear()

			ts1_list = []
			for tree in ts1.trees():
				for site in tree.sites():
					f = tree.get_num_leaves(site.mutations[0].node) / n_haps1 # allele frequency
					if f > args.maf and f < 1-args.maf:
						ts1_list.append(site)
#
			ts2_list = []
			for tree in ts2.trees():
				for site in tree.sites():
					f = tree.get_num_leaves(site.mutations[0].node) / n_haps2 # allele frequency
					if f > args.maf and f < 1-args.maf:
						ts2_list.append(site)
			both_list = [] # list of sites common in both tree sequences
			for site1 in ts1_list:
				for site2 in ts2_list:
					if site2.position > site1.position: # speed up that assumes positions are ordered from small to large
						break
					if site1.position==site2.position and site1.ancestral_state==site2.ancestral_state:
						both_list.append((site1, site2))

			for site1, site2 in both_list:
				common_site_id1 = tables1.sites.add_row(
					position=site1.position,
					ancestral_state=site1.ancestral_state)
				tables1.mutations.add_row(
					site=common_site_id1,
					node=site1.mutations[0].node,
					derived_state=site1.mutations[0].derived_state)
				common_site_id2 = tables2.sites.add_row(
					position=site2.position,
					ancestral_state=site2.ancestral_state)
				tables2.mutations.add_row(
					site=common_site_id2,
					node=site2.mutations[0].node,
					derived_state=site2.mutations[0].derived_state)
			ts_list1[chr_idx] = tables1.tree_sequence()
			ts_list2[chr_idx] = tables2.tree_sequence()

		return ts_list1, ts_list2

def run_gwas(args, y, ts_list_geno, m_geno_total):
	r'''
	Get GWAS beta-hats
	'''
	betahat_A_list = [None for i in range(args.n_chr)] # list of np arrays (one for each chromosome) holding GWAS beta-hats
	maf_A_list = [None for i in range(args.n_chr)] # list of np arrays (one for each chromosome) holding GWAS MAF
#	 pval_A_list = [None]*args.n_chr # list of np arrays (one for each chromosome) holding GWAS pvals

	n = int(ts_list_geno[0].get_sample_size()/2 ) # assume that sample size is same across chromosomes
	for chr_idx in range(args.n_chr):
		m_gwas = ts_list_geno[chr_idx].get_num_mutations()
		betahat_A = np.empty(shape=m_gwas)
		maf_A = np.empty(shape=m_gwas)
#		 pval_A = np.empty(shape=m_geno_total)
		print(f'Determining beta-hats in chromosome {chr_idx+1}')
		for k, variant in enumerate(ts_list_geno[chr_idx].variants()):
				X_A = nextSNP_add(variant, index=None)
				betahat, _, _, pval, _ = stats.linregress(x=X_A, y=y.reshape(n,))
				betahat_A[k] = betahat
				af = variant.genotypes.astype(int).mean()
				maf = min(af, 1-af)
				maf_A[k] = maf
#				 pval_A[k] = pval
		betahat_A_list[chr_idx] = betahat_A
		maf_A_list[chr_idx] = maf_A
#		 pval_A_list[chr_idx] = pval_A

	return betahat_A_list, maf_A_list

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
			 chrom, beta_std, seed):
		print('... MCMC ...')

		# seed
		if seed != None:
			random.seed(seed)

		# derived stats
		beta_mrg = np.array(sst_dict['BETA']).T
		beta_mrg = np.expand_dims(beta_mrg, axis=1)
		maf = np.array(sst_dict['MAF']).T
		n_pst = (n_iter-n_burnin)/thin
#		 p = len(sst_dict['SNP'])
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
			if itr % 100 == 0:
				print('--- iter-' + str(itr) + ' ---')

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

#		 # write posterior effect sizes
#		 if phi_updt == True:
#			 eff_file = out_dir + '_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
#		 else:
#			 eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)
#
#		 with open(eff_file, 'w') as ff:
#			 for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
#				 ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta))

		# print estimated phi
		if phi_updt == True:
			print('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

		print('... Done ...')
		return beta_est

	a = 1; b = 0.5
	phi = None
	n = args.n
	n_iter = 1000
	n_burnin = 500
	thin = 5
	beta_std = True
	seed = None

	sst_dict_list = [{'BETA':betahat_A_list[chr_idx], 'MAF':maf_A_list[chr_idx]}
					 for chr_idx in range(args.n_chr)]

	beta_est_list = []
	for chr_idx in range(args.n_chr):
		sst_dict = sst_dict_list[chr_idx]
		ld_blk = ld_list[chr_idx]
		blk_size = [len(blk) for blk in ld_blk]
		beta_est_list.append(mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size,
								  n_iter, n_burnin, thin, chr_idx+1, beta_std, seed))
	return beta_est_list

if __name__ == '__main__':
	args = parser.parse_args()

	# simulate tree sequences
	start_sim_ts = dt.now()
	args, ts_list_all, ts_list_geno_all, m, m_start, m_total, m_geno, m_geno_start, \
	m_geno_total, n_pops, genotyped_list_index	= sim_ts(args)
	print(f'sim ts time (min): {round((dt.now()-start_sim_ts).seconds/60, 2)}')

	# split into ref and non-ref
	ts_list_ref, ts_list, ts_list_geno, m, m_start, m_total, m_geno, m_geno_start, \
	m_geno_total, genotyped_list_index = split(args, ts_list_all, ts_list_geno_all) # maf = -1 removes the MAF filter

	# simulate phenotype
	start_sim_phen = dt.now()
	ts_list0 = ts_list.copy()
	y, beta_A_list, ts_pheno_A_list, causal_A_idx_list	= sim_phen(args, n_pops, ts_list, m_total)
	# TODO: Check that individuals are in the same order in ts_pheno_A_list and ts_list
	print(len(causal_A_idx_list[0]))
	print(f'sim phen time: {round((dt.now()-start_sim_phen).seconds/60, 2)} min')

	# MAF filter ref and non-ref
	start_joint_maf = dt.now()
	ts_list_ref, ts_list = joint_maf_filter(ts_list1=ts_list_ref, ts_list2=ts_list,
											maf=args.maf)
	ts_list_geno, genotyped_list_index, m_total, m_geno_total = _update_vars(args, ts_list) # update only for discovery cohort
	print(f'> post maf filter variant ct: {m_total}')
	print(f'joint maf filter time: {round((dt.now()-start_joint_maf).seconds/60, 2)} min')

	causal_idx_pheno_list = [] # get indices of causal variants in MAF filtered dataset; type=list of lists
	for chr_idx in range(args.n_chr):
		ts_causal = ts_pheno_A_list[chr_idx]
		ts_geno = ts_list_geno[chr_idx]
		sites_causal = [site.position for tree in ts_causal.trees() for site in tree.sites()] # all sites with MAF>0
		sites_geno = [site.position for tree in ts_geno.trees() for site in tree.sites()]
		geno_index = [k for k, position in enumerate(sites_causal) if position in sites_geno]
		geno_index = np.asarray(geno_index)
		causal_idx_pheno_list.append(geno_index)

	causal_idx_list = [] # get indices of causal variants in MAF filtered dataset; type=list of lists
	for chr_idx in range(args.n_chr):
		ts_causal = ts_pheno_A_list[chr_idx]
		ts_geno = ts_list_geno[chr_idx]
		sites_causal = [site.position for tree in ts_causal.trees() for site in tree.sites()] # all sites with MAF>0
		sites_geno = [site.position for tree in ts_geno.trees() for site in tree.sites()]
		geno_index = [k for k, position in enumerate(sites_geno) if position in sites_causal]
		geno_index = np.asarray(geno_index)
		causal_idx_list.append(geno_index)

	# run GWAS (and calculate MAF along the way)
	start_run_gwas = dt.now()
	betahat_A_list, maf_A_list = run_gwas(args, y, ts_list_geno, m_geno_total)
	print(f'run gwas time: {round((dt.now()-start_run_gwas).seconds/60, 2)} min')

	for chr_idx in range(args.n_chr):
		causal_idx_phen = causal_idx_pheno_list[chr_idx]
		causal_idx = causal_idx_list[chr_idx]
		beta_A_pheno = np.zeros(shape=len(betahat_A_list[chr_idx]))
		beta_A_pheno[causal_idx] = beta_A_list[chr_idx][causal_idx_phen]
#		beta_A_geno = beta_A_list[chr_idx][geno_index_list[chr_idx]]
		r = np.corrcoef(np.vstack((beta_A_pheno, betahat_A_list[chr_idx])))[0,1]
		print(f'correlation between betas: {r}') #subset to variants that were used in the GWAS

	# calculate LD matrix
	start_calc_ld = dt.now()
	ld_list = calc_ld(args, ts_list_ref)
	print(f'calc ld time: {round((dt.now()-start_calc_ld).seconds/60, 2)} min')

	# run PRS-CS
	start_prs_cs = dt.now()
	beta_est_list = prs_cs(args, betahat_A_list, maf_A_list, ld_list)
	print(f'prs-cs time: {round((dt.now()-start_prs_cs).seconds/60, 2)} min')

	print(beta_est_list[0].shape)

	for chr_idx in range(args.n_chr):
		causal_idx_phen = causal_idx_pheno_list[chr_idx]
		causal_idx = causal_idx_list[chr_idx]
		beta_A_pheno = np.zeros(shape=len(betahat_A_list[chr_idx]))
		beta_A_pheno[causal_idx] = beta_A_list[chr_idx][causal_idx_phen]
		beta_est = np.squeeze(beta_est_list[chr_idx])
		r = np.corrcoef(np.vstack((beta_A_pheno, beta_est)))[0,1]
		print(f'correlation between betas: {r}') #subset to variants that were used in the GWAS

	n = int(ts_list_geno[0].get_sample_size()/2 )
	yhat = np.zeros(n)
	for chr_idx in range(args.n_chr):
		ts_geno = ts_list_geno[chr_idx]
		beta_est = np.squeeze(beta_est_list[chr_idx])
		for k, variant in enumerate(ts_geno.variants()): # Note, progress here refers you to tqdm which just creates a pretty progress bar.
			X_A = nextSNP_add(variant)
			yhat += X_A * beta_est[k]

	r = np.corrcoef(np.vstack((y, yhat)))[0,1]
	print(f'y w/ yhat correlation: {r}')
