from side_code.config import *
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
import logging
import random
import re
import pandas as pd
import re
from sklearn.decomposition import PCA

def get_msa_name(msa_path, general_msa_dir):
    return msa_path.replace(general_msa_dir, "").replace(os.path.sep,
                                                         "_")

def get_local_path(path):
    if LOCAL_RUN:
        return path.replace("/groups/pupko/noaeker/", "/Users/noa/Workspace/")
    else:
        return path


def get_msa_data(msa_path, msa_suffix):
    with open(msa_path) as original:
        msa_data = list(SeqIO.parse(original, msa_suffix))
    return msa_data


def extract_file_type(path, change_format=False, ete=False):
    filename, file_extension = os.path.splitext(path)
    if change_format:
        if file_extension == '.phy':
            file_extension = 'iphylip' if ete == True else 'phylip-relaxed'
        elif file_extension == ".fasta":
            file_extension = 'fasta'
        elif file_extension == ".nex":
            file_extension = 'nexus'
    return file_extension



def remove_gaps_and_trim_locis(sample_records, max_n_loci, loci_shift):
    all_data = np.array([list(record.seq) for record in sample_records])
    count_gaps_per_column = np.count_nonzero(((all_data == "-") | (all_data == "X")), axis=0)
    non_gapped_data = all_data[:, count_gaps_per_column < all_data.shape[0]]
    loci_trimmed_data = non_gapped_data[:, loci_shift:loci_shift + max_n_loci]
    new_sampled_records = []
    for i, old_record in enumerate(sample_records):
        sampled_record = SeqRecord(Seq("".join(list(loci_trimmed_data[i, :]))), id=old_record.id, name=old_record.name,
                                   description=old_record.description)
        new_sampled_records.append(sampled_record)
    return new_sampled_records


def bootstrap_MSA(msa_path,out_msa_path):
    all_msa_records = get_alignment_data(msa_path)
    seq_size = len(str(all_msa_records[0].seq))
    sample = random.choices(list(range(seq_size)), k=(seq_size))
    new_msa_records  = []
    for record in all_msa_records:
        id = record.id
        name =record.name
        description = record.description
        new_seq = ""
        for j in range(len(sample)):
            new_seq += str(record.seq)[sample[j]]
        sampled_record = SeqRecord(Seq(new_seq), id=id,
                                   name=name,
                                   description=description)
        new_msa_records.append(sampled_record)
    SeqIO.write(new_msa_records, out_msa_path, 'phylip-relaxed')
    return out_msa_path




def trim_n_seq(original_seq_records, number_of_sequences, seed):
    seq_trimmed_seq_records = []
    seq_values = set()
    random.seed(seed)
    random.shuffle(original_seq_records)
    for record in original_seq_records:
        if len(seq_trimmed_seq_records) >= number_of_sequences:
            break
        if str(record.seq) in seq_values:
            continue
        else:
            sampled_record = SeqRecord(record.seq, id=record.id, name=record.name,
                                       description=record.description)
            seq_values.add(str(record.seq))
            seq_trimmed_seq_records.append(sampled_record)
    return seq_trimmed_seq_records


def count_unique_n_seq(original_seq_records):
    seq_values = set()
    for record in original_seq_records:
        seq = np.array(list(record.seq))
        undetermined_deq = seq[(seq == "-") | (seq == "X")]
        if len(undetermined_deq) < len(seq):
            seq_values.add("".join(seq))
    return len(seq_values)


def trim_MSA(original_alignment_path, trimmed_alignment_path, number_of_sequences,max_n_loci, loci_shift):
    original_alignment_data = get_alignment_data(original_alignment_path)
    obtained_n_seq = -1
    i = 0
    while obtained_n_seq < number_of_sequences and i <= 100:
        seq_trimmed_seq_records = trim_n_seq(original_alignment_data, number_of_sequences, seed=SEED + i)
        loci_trimmed_seq_records = remove_gaps_and_trim_locis(seq_trimmed_seq_records, max_n_loci, loci_shift)
        obtained_n_seq = count_unique_n_seq(loci_trimmed_seq_records)
        i = i + 1
    logging.debug("obtained {obtained_n_seq} sequences after {i} iterations!".format(obtained_n_seq=obtained_n_seq, i=i))
    try:
        SeqIO.write(loci_trimmed_seq_records, trimmed_alignment_path, 'fasta')
        logging.debug(" {} sequences written succesfully to new file {}".format(len(seq_trimmed_seq_records),
                                                                               trimmed_alignment_path))
    except:
        logging.error("ERROR! {} sequences NOT written succesfully to new file {}".format(number_of_sequences,
                                                                                          trimmed_alignment_path))



def remove_MSAs_with_not_enough_seq(file_path_list, min_seq):
    proper_file_path_list = []
    for path in file_path_list:
        file_type_biopython = extract_file_type(path, True)
        with open(path) as file:
            n_seq = len(list(SeqIO.parse(file, file_type_biopython)))
            if n_seq >= min_seq:
                proper_file_path_list.append(path)
    return proper_file_path_list



def get_positions_stats(alignment_df):
    alignment_df_fixed = alignment_df.replace('-', np.nan)
    gap_positions_pct = np.mean(alignment_df_fixed.isnull().sum() / len(alignment_df_fixed))
    counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
    probabilities = [list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
                     for col in
                     list(alignment_df)]
    entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
    avg_entropy = np.mean(entropy)
    constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
    return constant_sites_pct, avg_entropy, gap_positions_pct


def get_alignment_data(msa_path):

    try:
        with open(msa_path) as file:
            data = list(SeqIO.parse(file, 'fasta'))
            if len(data)==0:
                raise Exception("zero value")
    except:
        try:
            with open(msa_path) as file:
                data = list(SeqIO.parse(file, 'phylip-relaxed'))
        except:
                    return -1

    return data


def change_sequence_names(msa_path,out_msa_path):
    all_msa_records = get_alignment_data(msa_path)
    new_msa_records = []
    for i,record in enumerate(all_msa_records):
        sampled_record = SeqRecord(record.seq, id=f"seq_{i}",
                                   name=f"seq_{i}",
                                   description=f"seq_{i}")
        new_msa_records.append(sampled_record)
    SeqIO.write(new_msa_records, out_msa_path, 'phylip-relaxed')
    return out_msa_path


def alignment_list_to_df(alignment_data):
    alignment_list = [list(alignment_data[i].seq) for i in range(len(alignment_data))]
    loci_num = len(alignment_data[0].seq)
    columns = list(range(0, loci_num))
    original_alignment_df = pd.DataFrame(alignment_list, columns=columns)
    return original_alignment_df

def remove_env_path_prefix(path):
    path = path.replace("/groups/pupko/noaeker/", "")
    path = path.replace("/Users/noa/Workspace/","")
    return  path


def remove_MSAs_with_not_enough_seq_and_locis(file_path_list, min_n_seq, max_n_seq, min_n_loci):
    proper_file_path_list = []
    for path in file_path_list:
        data = get_alignment_data(path)
        if data==-1:
            continue
        n_seq = len(data)
        n_loci = len(data[0])
        if n_seq >= min_n_seq and n_seq<=max_n_seq and n_loci>= min_n_loci:
            proper_file_path_list.append(path)
    return proper_file_path_list


def concatenate_MSAs(msa_path_list, out_msa_path):
    all_msas_records = []
    for msa_path in msa_path_list:
        with open(msa_path) as file:
            file_type_biopython = extract_file_type(msa_path, True)
            sample_records = list(SeqIO.parse(file, file_type_biopython))
            all_msas_records.append(sample_records)
    new_msa_records = []
    for i in range(len(all_msas_records[0])): #iterating over number of sequences
        new_seq = ""
        id = all_msas_records[0][i].id
        name = all_msas_records[0][i].name
        description = all_msas_records[0][i].name
        for j in range(len(all_msas_records)):
            new_seq += str(all_msas_records[j][i].seq)
        sampled_record = SeqRecord(Seq(new_seq), id=id,
                                   name=name,
                                   description=description)
        new_msa_records.append(sampled_record)
    SeqIO.write(new_msa_records, out_msa_path, 'phylip-relaxed')
    return out_msa_path


def main():
    msas_path_list = ['/Users/noa/Downloads/FNA2AA-upp-masked-c12.fasta']
    change_sequence_names(msas_path_list[0], out_msa_path='/Users/noa/Workspace/sim_0_new.phy')
    #concatenate_MSAs(msa_path_list=msas_path_list, out_msa_path='/Users/noa/Workspace/simulations_results/raxml_grove_simulations/job_0/unified_MSA.phy')


if __name__ == "__main__":
    main()


