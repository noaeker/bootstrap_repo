import sys

if sys.platform == "linux" or sys.platform == "linux2":

    PROJECT_ROOT_DIRECRTORY = "/groups/pupko/noaeker/RAxML_params_tuning"
else:
    PROJECT_ROOT_DIRECRTORY = "/Users/noa/Workspace/RAxML_params_tuning"
sys.path.append(PROJECT_ROOT_DIRECRTORY)

from side_code.basic_trees_manipulation import *
from side_code.MSA_manipulation import get_alignment_data, alignment_list_to_df, get_local_path
from side_code.config import *
from pypythia.predictor import DifficultyPredictor
from pypythia.prediction import get_all_features
from pypythia.raxmlng import RAxMLNG
from pypythia.msa import MSA
import numpy as np


def pct_25(values):
    return np.percentile(values, 25)


def pct_75(values):
    return np.percentile(values, 75)

def get_summary_statistics_dict(feature_name, values, funcs={'mean': np.mean, 'median': np.mean, 'var': np.var,
                                                             'pct_25': pct_25, 'pct_75': pct_75,
                                                             'min': np.min, 'max': np.max,
                                                             }):
    res = {}
    for func in funcs:
        res.update({f'{feature_name}_{func}': (funcs[func])(values)})
    return res


def pypythia(msa_path, model):
    predictor = DifficultyPredictor(open(f"{PROJECT_ROOT_DIRECRTORY}/pypythia/predictor.pckl", "rb"))
    raxmlng = RAxMLNG(RAXML_NG_EXE)
    msa = MSA(msa_path)
    msa_features = get_all_features(raxmlng, msa, model)
    difficulty = predictor.predict(msa_features)
    return difficulty


def get_msa_stats(msa_path, model):
    msa_data = get_alignment_data(get_local_path(msa_path))
    n_seq, n_loci = len(msa_data), len(msa_data[0].seq)
    all_msa_features = {"feature_msa_n_seq": n_seq, "feature_msa_n_loci": n_loci, "msa_path": msa_path,
                       }
    alignment_data = get_alignment_data(get_local_path(msa_path))
    alignment_df = alignment_list_to_df(alignment_data)
    alignment_df_fixed = alignment_df.replace('-', np.nan)
    alignment_df_unique = alignment_df.T.drop_duplicates().T
    counts_per_position = [dict(alignment_df_fixed[col].value_counts(dropna=True)) for col in list(alignment_df)]
    probabilities = [list(map(lambda x: x / sum(counts_per_position[col].values()), counts_per_position[col].values()))
                     for col in
                     list(alignment_df)]
    entropy = [sum(list(map(lambda x: -x * np.log(x), probabilities[col]))) for col in list(alignment_df)]
    constant_sites_pct = sum([1 for et in entropy if et == 0]) / len(entropy)
    try:
        msa_difficulty = pypythia(get_local_path(msa_path), model)
    except Exception:
        msa_difficulty = -1
    all_msa_features.update({"feature_msa_constant_sites_pct": constant_sites_pct,
                             "feature_msa_n_unique_sites": len(alignment_df_unique.columns),
                             "feature_msa_pypythia_msa_difficulty": msa_difficulty})

    multi_dimensional_features = {"feature_msa_entropy": entropy,
                                  }

    for feature in multi_dimensional_features:
        all_msa_features.update(get_summary_statistics_dict(feature, multi_dimensional_features[feature]))
    return all_msa_features