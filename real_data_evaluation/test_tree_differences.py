import pandas as pd
from ete3 import Tree
import numpy as np


def extract_trees_rf(data):
    rf_distances = []
    for i,row in data.iterrows():
        tree_a = Tree(row["final_tree_topology_path"], format = 0)
        tree_b = Tree(row["final_tree_aLRT"], format = 0)
        rf_distance = tree_a.robinson_foulds(tree_b)
        rf_distances.append(rf_distance)
    print(np.mean(rf_distances))
    print(np.median(rf_distances))



def main():
    doro_path = "/groups/pupko/noaeker/bootstrap_repo/real_data_evaluation/real_data_topologies_comparison_doro.csv"
    dna_path = "/groups/pupko/noaeker/bootstrap_repo/real_data_evaluation/real_data_topologies_comparison.csv"
    extract_trees_rf(doro_path)
    extract_trees_rf(dna_path)


if __name__ == "__main__":
    main()