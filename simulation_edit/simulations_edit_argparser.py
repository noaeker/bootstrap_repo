import argparse
def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_fasttree', type=str,
                        default="/Users/noa/Workspace/bootstrap_results/test2/job_0/simulations_df_fasttree.tsv")
    parser.add_argument('--data_path_raxml', type=str,
                        default="/Users/noa/Workspace/bootstrap_results/test2/job_0/simulations_df_raxml.tsv")
    parser.add_argument('--data_path_iqtree', type=str,
                        default="/Users/noa/Workspace/bootstrap_results/test2/job_0/simulations_df_iqtree.tsv")
    parser.add_argument('--work_path', type=str,
                        default='/Users/noa/Workspace/bootstrap_results/bootstrap_edit_results')
    parser.add_argument('--n_pars', type=int, default=50)
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--jobs_prefix', type=str, default = 'test_boot_edit')
    parser.add_argument('--queue', type = str, default = 'power-pupko')
    return parser


def job_parser():
    parser = main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--job_data_path', action='store', type=str)
    parser.add_argument('--job_work_path', action='store', type=str)
    parser.add_argument('--job_final_output_path',action='store', type=str)
    return parser