import math
import msprime
import gzip
from datetime import datetime as dt
import subprocess
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--n', type=int, required=True, help="Number of individuals to simulate")
parser.add_argument('--m', type=int, required=True, help="Length of the region analysed in nucleotides")
parser.add_argument('--rec', type=float, required=False, default=2e-8, help="Recombination rate")
parser.add_argument('--mut', type=float, required=False, default=2e-8, help="Mutation rate")
parser.add_argument('--maf', type=float, required=False, default=0.05, help="MAF to filter variants by")
parser.add_argument('--plink', type=str, required=False, default='/home/nbaya/plink', help='path to plink executable')
args = parser.parse_args()

n = args.n
m = args.m
rec = args.rec
mut = args.mut
maf = args.maf
plink = args.plink

def out_of_africa(N_haps, no_migration):
	N_A = 7300
	N_B = 2100
	N_AF = 12300
	N_EU0 = 1000
	N_AS0 = 510
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
		m_AF_B = 0
		m_AF_EU = 0
		m_AF_AS = 0
		m_EU_AS = 0
	else:
		m_AF_B = 25e-5
		m_AF_EU = 3e-5
		m_AF_AS = 1.9e-5
		m_EU_AS = 9.6e-5
	
	# Population IDs correspond to their indexes in the population
	# configuration array. Therefore, we have 0=YRI, 1=CEU and 2=CHB
	# initially.
	n_pops = 3

	population_configurations = [
		msprime.PopulationConfiguration(sample_size=2*N_haps[0], initial_size=N_AF),
		msprime.PopulationConfiguration(sample_size=2*N_haps[1], initial_size=N_EU, growth_rate=r_EU),
		msprime.PopulationConfiguration(sample_size=2*N_haps[2], initial_size=N_AS, growth_rate=r_AS)
		]
	
	migration_matrix = [[0, m_AF_EU, m_AF_AS],
						[m_AF_EU, 0, m_EU_AS],
						[m_AF_AS, m_EU_AS, 0],
						]
	
	demographic_events = [
	# CEU and CHB merge into B with rate changes at T_EU_AS
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
	return population_configurations, migration_matrix, demographic_events, N_A, n_pops, N_EU

N_sim = n


N_haps = [0, int(N_sim/2) , 0]
population_configurations, migration_matrix, demographic_events, N_A, n_pops, N_EU  = out_of_africa(N_haps=N_haps, no_migration=False)


print(f'EU size: {N_EU}')

## Simulate

Ne = N_EU
out = f'test_msprime.Nsim_{N_sim}.m_{m}.maf_{maf}'

print(f'\n... Starting simulation ...\nN_sim={N_sim}\tm={m}\nout: {out}')
start_sim = dt.now()
print('Time: {:%H:%M:%S (%Y-%b-%d)}'.format(start_sim))

startfile = open(out+'.starttime.txt','w')
startfile.write(f'starttime: {start_sim}\nNsim\tm\tmaf\n{N_sim}\t{m}\t{maf}')
startfile.close()

ts = msprime.simulate(Ne = Ne, 
                      length = m,
                      recombination_rate=rec,
                      mutation_rate=mut,
                      population_configurations=population_configurations,
                      demographic_events=demographic_events) 

elapsed_ts = dt.now()-start_sim
print(f'\n... Elapsed time for ts generation: {round(elapsed_ts.seconds/60, 2)} min ...')

start_maf = dt.now()
print(f'\n... Starting MAF filter (MAF>{maf}) ...\ntime: {start_maf}')

def get_common_mutations(maf, tree):
	ps = tree_sequence.get_sample_size()
	log.log('Determining sites > MAF cutoff {m}'.format(m=maf))

	tables = tree_sequence.dump_tables()
	tables.mutations.clear()
	tables.sites.clear()

	for tree in tree_sequence.trees():
		for site in tree.sites():
			f = tree.get_num_leaves(site.mutations[0].node) / n_haps
			if f > maf and f < 1-maf:
				common_site_id = tables.sites.add_row(
					position=site.position,
					ancestral_state=site.ancestral_state)
				tables.mutations.add_row(
					site=common_site_id,
					node=site.mutations[0].node,
					derived_state=site.mutations[0].derived_state)
	new_tree_sequence = tables.tree_sequence()

elapsed_maf = dt.now()-start_maf

print(f'\n... Elapsed time for MAF filter (MAF>{maf}) ...\nelapsed time: {round(elapsed_maf.seconds/60, 2)} min')

start_vcf = dt.now()
print('\n... Starting to write VCF ...\ntime: {:%H:%M:%S (%Y-%b-%d)}'.format(start_vcf))

with gzip.open(f'{out}.vcf.gz','wt') as f:
    ts.write_vcf(f)

elapsed_vcf = dt.now()-start_vcf
print(f'... Elapsed time for VCF creation: {round(elapsed_vcf.seconds/60, 2)} min ...')

start_plink = dt.now()
print('\n... Starting to convert to PLINK format ...\ntime: {:%H:%M:%S (%Y-%b-%d)}'.format(start_plink))
subprocess.call([plink, '--vcf', f'{out}.vcf.gz', '--double-id','--make-bed','--out', out])

elapsed_plink = dt.now()-start_plink
print(f'... Elapsed time for VCF -> PLINK conversion: {round(elapsed_plink.seconds/60, 2)} min ...')


elapsed_total = dt.now()-start_sim
print(f'... Total elapsed time for simulation: {round(elapsed_total.seconds/60, 2)} min ...') 

subprocess.call(['rm', f'{out}.vcf.gz'])
