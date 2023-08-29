import argparse
def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_trees', action='store', type=int, default=10)
    parser.add_argument('--number_of_MSAs_per_tree', type=int, default=1)
    parser.add_argument('--min_n_taxa', type = int, default = 5)
    parser.add_argument('--max_n_taxa', type=int, default=10)
    parser.add_argument('--min_n_loci', type=int, default=100)
    parser.add_argument('--max_n_loci',type=int,default = 200)
    parser.add_argument('--n_jobs', type = int, default= 1)
    parser.add_argument('--run_prefix', type = str, default = 'test4')
    parser.add_argument('--jobs_prefix', type = str, default='RG')
    parser.add_argument('--waiting_time_between_job_submissions',type=int, default =4)
    parser.add_argument('--safe_run', action='store_true', default= True)
    parser.add_argument('--queue', type = str, default = 'power-pupko')
    parser.add_argument('--n_cpus_per_job', type = int, default= 1)
    parser.add_argument('--msa_type', type = str, default= 'DNA')
    parser.add_argument('--nb_iqtree',type = int, default= 1000)
    parser.add_argument('--nb_fasttree', type=int, default=10)
    parser.add_argument('--calc_features', action='store_true', default=True)
    return parser


def job_parser():
    parser = main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--curr_job_folder', action='store', type=str)
    return parser