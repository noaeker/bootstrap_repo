import argparse

def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_alignments_folder', type=str,
                        default="/Users/noa/Workspace/bootstrap_results/msa_example")
    parser.add_argument('--name', type=str,
                        default="real_data_eval")
    parser.add_argument('--n_cpus', type = int, default = 2)
    parser.add_argument('--jobs_prefix', type=str, default = 'test_boot_edit')
    parser.add_argument('--queue', type = str, default = 'power-pupko')
    parser.add_argument('--specific_study', type= str, default= 'Bergsten_2013')
    parser.add_argument('--change_names',action='store_true', default=False)
    parser.add_argument('--model',  type = str, default = 'WAG+G')
    parser.add_argument('--use_existing_trees', action='store_true', default=False)
    parser.add_argument('--program',type=str, default = 'iqtree')
    return parser


def job_parser():
    parser = main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--job_folder', action='store', type=str)
    parser.add_argument('--job_study_name', action='store', type=str)
    return parser