import sys
import sys
if sys.platform == "linux" or sys.platform == "linux2":
    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/bootstrap_repo"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/bootstrap_repo"
sys.path.append(PROJECT_ROOT_DIRECRTORY)



from side_code.config import *
from side_code.code_submission import submit_linux_job, submit_local_job, generate_argument_list,generate_argument_str
from side_code.file_handling import create_dir_if_not_exists, create_or_clean_dir
from simulations_generation.simulations_argparser import main_parser
import os
import time

def generate_results_folder(curr_run_prefix):
    create_dir_if_not_exists(RESULTS_FOLDER)
    curr_run_prefix = os.path.join(RESULTS_FOLDER, curr_run_prefix)
    create_or_clean_dir(curr_run_prefix)
    return curr_run_prefix





def submit_single_job(all_jobs_results_folder, job_ind,args):
    curr_job_folder = os.path.join(all_jobs_results_folder, "job_" + str(job_ind))
    create_or_clean_dir(curr_job_folder)
    curr_job_log_path = os.path.join(curr_job_folder, str(job_ind) + "_tmp_log")
    run_command = f' python {MAIN_CODE_PATH_SIM} --job_ind {job_ind} --curr_job_folder {curr_job_folder}   {generate_argument_str(args)} '
    job_name = f"{job_ind}_{args.jobs_prefix}"
    if not LOCAL_RUN:
        submit_linux_job(job_name, curr_job_folder, curr_job_log_path, run_command, args.n_cpus_per_job, job_ind,
                         queue=args.queue)
    else:
        submit_local_job(MAIN_CODE_PATH_SIM,
                         ["--job_ind", str(job_ind), "--curr_job_folder", curr_job_folder,
                          ] + generate_argument_list(args))



def main():
    parser = main_parser()
    args = parser.parse_args()
    all_jobs_results_folder = generate_results_folder(args.run_prefix)
    all_jobs_general_log_file = os.path.join(all_jobs_results_folder, "log_file.log")
    logging.basicConfig(filename=all_jobs_general_log_file, level=logging.INFO)
    for job_ind in range((args.n_jobs)):
        logging.info(f"Submitted job number {job_ind}")
        submit_single_job(all_jobs_results_folder, job_ind, args)
        time.sleep(args.waiting_time_between_job_submissions)

if __name__ == "__main__":
    main()
