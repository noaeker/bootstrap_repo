import argparse
def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default="/Users/noa/Workspace/bootstrap_results/test2/job_0/simulations_df_fasttree.tsv")
    parser.add_argument('--final_output_path', type=str,
                        default="total_data.tsv")
    parser.add_argument('--work_path', type=str,
                        default='/Users/noa/Workspace/bootstrap_results/bootstrap_edit_results')
    parser.add_argument('--n_pars', type=int, default=50)
    parser.add_argument('--program', type=str, default='fasttree')
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--jobs_prefix', type=str, default = 'test_boot_edit')
    return parser


def job_parser():
    parser = main_parser()
    parser.add_argument('--job_ind', action='store', type=int)
    parser.add_argument('--job_data_path', action='store', type=str)
    parser.add_argument('--job_work_path', action='store', type=str)
    parser.add_argument('--job_final_output_path',action='store', type=str)
    return parser