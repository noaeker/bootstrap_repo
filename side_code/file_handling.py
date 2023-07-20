from side_code.config import *
import os
import pandas as pd
import shutil
import logging



def delete_file_content(file_path):
    with open(file_path, 'w'):
        pass


def extract_alignment_files_from_dirs(general_dir_path):
    files_list = []
    if os.path.exists(general_dir_path):
        for sub_dir in os.listdir(general_dir_path):
            sub_dir_path = os.path.join(general_dir_path, sub_dir)
            if os.path.isdir(sub_dir_path):
                for file in os.listdir(sub_dir_path):
                    files_list.append(os.path.join(sub_dir_path, file))
    return files_list


def alignment_list_to_df(alignment_data):
    alignment_list = [list(alignment_data[i].seq) for i in range(len(alignment_data))]
    loci_num = len(alignment_data[0].seq)
    columns = list(range(0, loci_num))
    original_alignment_df = pd.DataFrame(alignment_list, columns=columns)
    return original_alignment_df


def delete_dir_content(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            logging.error('Failed to delete %s. Reason: %s' % (file_path, e))
            return False
    return True


def create_or_clean_dir(dir):
    if os.path.exists(dir):
        delete_dir_content(dir)
    else:
        os.mkdir(dir)


def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def unify_text_files(input_path_list, output_file_path, str_given=False):
    with open(output_file_path, 'w') as outfile:
        if str_given:
            for input in input_path_list:
                outfile.write(input)
        else:
            for fname in input_path_list:
                with open(fname) as infile:
                    outfile.write(infile.read())
    return output_file_path


def add_csvs_content_to_current_content(csvs_path_list, unified_csv_path):
    existing_df = [pd.read_csv(unified_csv_path, sep=CSV_SEP)] if os.path.exists(unified_csv_path) else []
    existing_df_size = pd.read_csv(unified_csv_path, sep=CSV_SEP).size if os.path.exists(unified_csv_path) else 0
    logging.info('Existing df size is: {}'.format(existing_df_size))
    non_empty_df = [pd.read_csv(f, sep=CSV_SEP) for f in csvs_path_list if not pd.read_csv(f, sep=CSV_SEP).empty]
    combined_df = pd.concat(non_empty_df + existing_df, sort=False)
    combined_df_size = combined_df.size
    logging.info('Combined df size is: {}'.format(combined_df_size))
    combined_df.to_csv(unified_csv_path, sep=CSV_SEP)
    return combined_df


def add_csvs_content(csvs_path_list, unified_csv_path):
    non_empty_df = [pd.read_csv(f, sep=CSV_SEP) for f in csvs_path_list if not pd.read_csv(f, sep=CSV_SEP).empty]
    combined_df = pd.concat(non_empty_df, sort=False)
    combined_df_size = combined_df.size
    logging.info('Combined df size is: {}'.format(combined_df_size))
    combined_df.to_csv(unified_csv_path, sep=CSV_SEP)
    return combined_df


def remove_empty_columns(csv_path):
    if os.path.exists((csv_path)):
        df = pd.read_csv(csv_path, sep=CSV_SEP)
        df = df.dropna(how='all', axis=1)
        df.to_csv(csv_path, index=False, sep=CSV_SEP)


def unify_dicts(dicts):
    super_dict = {}
    for d in dicts:
        for k, v in d.items():
            super_dict[k] = v
    return super_dict


def extract_dir_list_from_csv(dir_list_csv_path):
    df = pd.read_csv(dir_list_csv_path)
    dir_list = [os.path.join(CSV_MSAs_FOLDER, path) for path in list(df["path"])]
    logging.debug("Number of paths in original csv = {n_paths}".format(n_paths=len(df.index)))
    return dir_list


def extract_alignment_files_from_general_csv(dir_list_csv_path):
    files_list = []
    logging.debug("Extracting alignments from {}".format(dir_list_csv_path))
    dir_list = extract_dir_list_from_csv(dir_list_csv_path)
    for dir in dir_list:
        if os.path.exists(dir):
            for file in os.listdir(dir):
                if (file.endswith(".phy") or file.endswith(".fasta")):
                    files_list.append(os.path.join(dir, file))
                    break
        else:
            logging.error("Following MSA dir does not exist {dir}".format(dir=dir))
    logging.debug("Overalls number of MSAs found in the given directories is: {nMSAs}".format(nMSAs=len(files_list)))
    return files_list