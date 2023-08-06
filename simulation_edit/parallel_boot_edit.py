import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from side_code.config import *
from side_code.code_submission import generate_argument_str, submit_linux_job, generate_argument_list, submit_local_job, execute_command_and_write_to_log
from simulation_edit.simulations_edit_argparser import main_parser
import pandas as pd
import os
import numpy as np
import time


def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_or_clean_dir(curr_run_prefix)
    return curr_run_prefix


def distribute_MSAS_over_jobs(raw_data, all_jobs_results_folder,args):
    job_dict = {}
    tree_names = list(np.unique(raw_data["tree_id"]))
    tree_splits = np.array_split(list(tree_names), min(args.n_jobs, len(tree_names)))
    for job_ind, job_trees in enumerate(tree_splits):
        logging.info(f"Job {job_ind} runs on {len(job_trees)} trees")
        time.sleep(10)
        curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
        create_or_clean_dir(curr_job_folder)
        current_raw_data_path = os.path.join(curr_job_folder, f"job_{job_ind}_raw_data{CSV_SUFFIX}")
        job_raw_data = raw_data.loc[raw_data.tree_id.isin(job_trees)]
        job_raw_data.to_csv(current_raw_data_path,sep=CSV_SEP)
        current_feature_output_path = os.path.join(curr_job_folder, f"job_raw_data_with_features{CSV_SUFFIX}")

        run_command = f' python {BOOTSTRAP_EDIT_CODE} --job_ind {job_ind} --job_work_path {curr_job_folder} --job_data_path {current_raw_data_path} --job_final_output_path {current_feature_output_path} {generate_argument_str(args)}'


        job_name = args.jobs_prefix + str(job_ind)
        if not LOCAL_RUN:
            curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
            submit_linux_job(job_name, curr_job_folder, curr_job_log_path, run_command, cpus=1,
                             job_ind=job_ind,
                             queue=args.queue)
        else:
            submit_local_job(BOOTSTRAP_EDIT_CODE,
                             ["--job_ind", str(job_ind), "--job_work_path", curr_job_folder, "--job_data_path",
                              current_raw_data_path,
                              "--job_final_output_path", current_feature_output_path,
                              ]+ generate_argument_list(args))
        job_dict[job_ind] = {"current_feature_output_path": current_feature_output_path, "job_name": job_name}
    return job_dict

def finish_all_running_jobs(job_names):
    logging.info("Deleting all jobs")
    for job_name in job_names: # remove all remaining folders
            delete_current_job_cmd = f"qstat | grep {job_name} | xargs qdel"
            execute_command_and_write_to_log(delete_current_job_cmd, print_to_log=True)


def main():
    parser = main_parser()
    args = parser.parse_args()
    feature_pipeline_dir = args.work_path
    create_or_clean_dir(feature_pipeline_dir)
    log_file_path = os.path.join(feature_pipeline_dir, "general_features.log")
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
    raw_data = pd.read_csv(args.data_path, sep = CSV_SEP)

    distribute_MSAS_over_jobs(raw_data, feature_pipeline_dir,args)


if __name__ == "__main__":
    main()
