import argparse
import os

def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default=os.getcwd())
    parser.add_argument('--RFE', action='store_true', default=False)
    parser.add_argument('--full_grid', action='store_true', default=False)
    parser.add_argument('--programs', type=str, default='raxml')
    parser.add_argument('--cpus_per_main_job', type=int, default=1)
    parser.add_argument('--sample_fracs', type=str, default='0.25_0.5_1')
    parser.add_argument('--inc_sample_fracs', action='store_true', default=False)
    parser.add_argument('--reunite_val_data', action='store_true', default=False)
    parser.add_argument('--use_val_data', action='store_true', default=True)
    parser.add_argument('--reunite_training_data', action='store_true', default=False)
    parser.add_argument('--main_data_folder', type=str,
                        default='/Users/noa/Workspace/bootstrap_results/remote_results/full_data')
    parser.add_argument('--validation_data_folder', type=str,
                        default='/Users/noa/Workspace/bootstrap_results/remote_results/all_val')
    parser.add_argument('--n_val_samp', type=int, default=250)
    parser.add_argument('--n_main_samp', type=int, default=6000)
    parser.add_argument('--ML_model', type=str, default='sgd')
    return parser