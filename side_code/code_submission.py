
from side_code.file_handling import create_dir_if_not_exists
from side_code.config import *
import os
import subprocess
import sys
import logging
import time
from subprocess import PIPE, STDOUT

def is_job_done(job_log_folder, started_file, job_start_time, timeout):
    if LOCAL_RUN:
        return True
    else:
        if not os.path.exists(started_file) and (time.time()-job_start_time)>timeout:
            logging.debug(f"Started file {started_file} does not appear after {time.time()-job_start_time} seconds, job will be terminated")
            return True
        for file in os.listdir(job_log_folder):
            if (file.endswith('.ER')):
                logging.debug(f"Found error log file= {file}")
                return True
    return False

def execute_command_and_write_to_log(command, print_to_log=True):
    if print_to_log:
        logging.debug(f"About to run: {command}")
    subprocess.run(command, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)

def generate_argument_list(args, exclude = []):
    output = []
    for arg in vars(args):
        if arg not in exclude:
            if not type(getattr(args, arg)) == bool:
                value = ["--" + arg, str(getattr(args, arg))]
            elif (getattr(args, arg)) == True:
                value = ["--" + arg]
            else:
                value = []
            output = output + value
        else:
            pass
    print(output)
    return output


def generate_argument_str(args, exclude = []):
    output = ""
    for arg in vars(args):
        if arg not in exclude:
            if not type(getattr(args, arg)) == bool:
                value = "--" + arg + " " + str(getattr(args, arg))
            elif (getattr(args, arg)) == True:
                value = "--" + arg
            else:
                value = ""
            output = output + value + " "
    return output.strip()

def submit_linux_job(job_name, job_folder,job_log_path, run_command, cpus, job_ind="job", queue='pupkolab'):
    create_dir_if_not_exists(job_folder)
    cmds_path = os.path.join(job_folder, str(job_ind) + ".cmds")
    job_line = f'{MODULE_LOAD_STR} cd {PROJECT_ROOT_DIRECRTORY}; {run_command}\t{job_name}'
    logging.debug("About to run on {} queue: {}".format(queue, job_line))
    with open(cmds_path, 'w') as cmds_f:
        cmds_f.write(job_line)
    command = f'{PBS_FILE_GENERATOR_CODE} {cmds_path} {job_log_path} --cpu {cpus} --q {queue}'
    logging.debug(f'About to submit a pbs file to {queue} queue based on cmds:{cmds_path}')
    subprocess.call(command, shell= True)


def submit_local_job(executable, argument_list):
    theproc = subprocess.Popen([sys.executable, executable] + argument_list)
    theproc.communicate()