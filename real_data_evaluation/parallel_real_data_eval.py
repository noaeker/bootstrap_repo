import sys

if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir, add_csvs_content
from side_code.config import *
from side_code.code_submission import generate_argument_str, submit_linux_job, generate_argument_list, submit_local_job, execute_command_and_write_to_log
from real_data_evaluation.real_data_evaluation_parsers import main_parser
from side_code.combine_current_results import unify_results_across_jobs
import pandas as pd
import os
import numpy as np
import time


def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_or_clean_dir(curr_run_prefix)
    return curr_run_prefix


def distribute_MSAS_over_jobs(study_names, all_jobs_results_folder,args):
    for job_ind, study in enumerate(study_names):
        logging.info(f"Job {job_ind} runs on study {study} study")
        time.sleep(10)
        curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
        create_dir_if_not_exists(curr_job_folder)
        run_command = f' python {REAL_DATA_ANALYSIS_CODE} --job_ind {job_ind} --job_folder {curr_job_folder} --job_study_name {study} {generate_argument_str(args)}'
        job_name = args.jobs_prefix + str(job_ind)
        if not LOCAL_RUN:
            curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
            submit_linux_job(job_name, curr_job_folder, curr_job_log_path, run_command, cpus=args.n_cpus,
                             job_ind=job_ind,
                             queue=args.queue)
        else:
            submit_local_job(REAL_DATA_ANALYSIS_CODE,
                             ["--job_ind", str(job_ind), "--job_folder", curr_job_folder, "--job_study_name",
                              study,
                              ]+ generate_argument_list(args))


def main():
    parser = main_parser()
    args = parser.parse_args()
    raw_results_folder = generate_results_folder(args.name)
    log_file_path = os.path.join(raw_results_folder, 'log_file.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    MSAs_stats_path = SUMMARY_FILE_PATH
    MSAs_stats = pd.read_csv(MSAs_stats_path)
    relevant_MSAs = MSAs_stats.loc[(MSAs_stats["dataset.alignment.ntax"]<=1000)&(MSAs_stats["dataset.alignment.nchar"]<10000)&(MSAs_stats["dataset.alignment.datatype"]=='nucleotide')]
    relevant_studies = relevant_MSAs["name"]
    logging.info(f"Relevant studies = {relevant_studies}")
    if args.specific_study is not None:
        relevant_studies = [args.specific_study]

    distribute_MSAS_over_jobs(relevant_studies, raw_results_folder,args)



if __name__ == "__main__":
    main()
