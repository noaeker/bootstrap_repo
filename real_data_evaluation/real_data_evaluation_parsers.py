import argparse

def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_alignments_folder', type=str,
                        default="/Users/noa/Workspace/bootstrap_results/msa_example")
    parser.add_argument('--name', type=str,
                        default="real_data_eval")
    parser.add_argument('--n_cpus', default = 2)
    parser.add_argument('--jobs_prefix', type=str, default = 'test_boot_edit')
    parser.add_argument('--queue', type = str, default = 'power-pupko')
    return parser


def job_parser():
    parser = main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--job_folder', action='store', type=str)
    parser.add_argument('--job_study_name', action='store', type=str)
    return parser