

from side_code.code_submission import execute_command_and_write_to_log
from side_code.config import *
from ete3 import *
import logging
import matplotlib as plt
import networkx as nx
import re
import shutil


class Edge:
    def __init__(self, node_a, node_b):
        self.node_a = node_a
        self.node_b = node_b

    def __str__(self):
        return ("[a={a} b={b}]".format(a=self.node_a, b=self.node_b))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if ((self.node_a == other.node_a) and (self.node_b == other.node_b)) or (
                (self.node_b == other.node_a) and (self.node_a == other.node_b)):
            return True
        else:
            return False


def print_subtree(tree, log_file, text):
    if log_file:
        log_file.write(text + " visualization: " + "\n" + tree.get_ascii(attributes=['name'],
                                                                         show_internal=True) + "\n")
        log_file.write(text + " newick " + str(tree.write(format=1)) + "\n")
    else:
        logging.info(text + " visualization: " + "\n" + tree.get_ascii(attributes=['name'], show_internal=True))
        logging.info(str(text + " newick " + tree.write(format=1)))


def add_internal_names(original_tree):
    for i, node in enumerate(original_tree.traverse()):
        if not node.is_leaf():
            node.name = "N{}".format(i)
        original_tree.get_tree_root().name = "ROOT"
    return original_tree


def generate_tree_object_from_newick(tree_nw, tree_type =0):
    starting_tree_object = Tree(newick=tree_nw, format=tree_type)
    add_internal_names(starting_tree_object)
    starting_tree_object.get_tree_root().name = "ROOT"
    return starting_tree_object


def generate_multiple_tree_object_from_newick_file(trees_path):
    with open(trees_path) as trees_path:
        newicks = trees_path.read().split("\n")
        newicks = [t for t in newicks if len(t) > 0]
        tree_objects = [generate_tree_object_from_newick(newick) for newick in newicks]
        return tree_objects


def generate_multiple_newicks_from_newick_file(trees_path):
    with open(trees_path) as trees_path:
        newicks = trees_path.read().split("\n")
        newicks = [t for t in newicks if len(t) > 0]
        return newicks


def generate_multiple_tree_object_from_newick_list(newicks ):
        tree_objects = [generate_tree_object_from_newick(newick) for newick in newicks]
        return tree_objects




def get_tree_string(tree_path):
    tree_object = Tree(newick=tree_path, format=1)
    return (tree_object.write(format=1))


def tree_branch_length_metrics(tree):
    internal_branch_lengths = []
    leaft_branch_lengths = []
    branch_lengths = []
    for node in tree.iter_descendants():
        # Do some analysis on node
        branch_lengths.append(node.dist)
        if node.is_leaf():
            leaft_branch_lengths.append(node.dist)
        else:
            internal_branch_lengths.append(node.dist)
    return {"BL_list": branch_lengths, "internal_BL_list": internal_branch_lengths, "leaf_BL_list": leaft_branch_lengths}


def assign_brlen_to_tree_object(tree_object, brlen_list):
    for i, node in enumerate(tree_object.iter_descendants()):
        # Do some analysis on node
        node.dist = brlen_list[i]
    return tree_object




def compute_largest_branch_length(tree):
    return max([node.dist for node in tree.iter_descendants()])


def get_distances_between_leaves(tree, topology_only = False):
    distances = []
    sorted_leaves = sorted([leaf.name for leaf in tree.iter_leaves()])
    for leaf_a_ind in range(len(sorted_leaves)):
        for leaf_b_ind in range(leaf_a_ind+1,len(sorted_leaves)):
                dist = tree.get_distance( sorted_leaves[leaf_a_ind], sorted_leaves[leaf_b_ind],topology_only= topology_only)
                distances.append(dist)
    return distances


def mad_tree_parameter(tree_path):
        mad_command = "{mad_exe_path} -t -s {tree_path}".format(mad_exe_path=MAD_COMMAND_PREFIX,
                                                                tree_path=tree_path)
        execute_command_and_write_to_log(mad_command)
        mad_log_path = tree_path + ".rooted"
        mad = extract_mad_file_statistic(mad_log_path)
        os.remove(tree_path)
        return mad

def extract_mad_file_statistic(mad_log_path):
    pattern = "MAD=([\d.]+)"
    with open(mad_log_path) as mad_output:
        data = mad_output.read()
        match = re.search(pattern, data, re.IGNORECASE)
    if match:
        value = float(match.group(1))
    else:
        error_msg = "Param  not found in mad file in {}".format(mad_log_path)
        logging.error(error_msg)
    return value


def translate_to_networkx_graph(tree):
    G = nx.Graph()
    for i, node in enumerate(tree.iter_descendants("levelorder")):
        if node.up:
            G.add_edge(node.name, node.up.name,length = node.dist)
    return G

def get_hyperbolic_tree_embeddings(curr_msa_stats, curr_data, partition_folder, i):
    relaxed_lasso = 1 if curr_msa_stats["relaxed_lasso"] else 0
    curr_data_path = os.path.join(partition_folder, f"partition_{i}_sitelh.csv")
    curr_data.to_csv(curr_data_path, index=False)
    command = f"module load R/3.6.1;Rscript --vanila {R_CODE_PATH} {curr_data_path} {partition_folder} {relaxed_lasso} {lasso_thresholds}"
    logging.info(f"About to run lasso command in glmnet: {command}")
    lasso_start_time = time.time()
    # os.system('module load R/3.6.1')
    os.system(command)
    logging.info("R glmnet command is done!")

    lasso_output_file_path = os.path.join(partition_folder, "r_lasso_relaxed.csv") if curr_msa_stats[
        "relaxed_lasso"] else os.path.join(partition_folder, "r_lasso.csv")
    glmnet_lasso_path = pd.read_csv(lasso_output_file_path)
    glmnet_running_time = time.time() - lasso_start_time
    logging.info(f"Lasso results should be found in : {lasso_output_file_path} ")
    return glmnet_lasso_path, glmnet_running_time


def main():
    t = Tree('((((H,K)D,(F,I)G)B,E)A,((L,(N,Q)O)J,(P,S)M)C);', format=1)
    add_internal_names(t)
    (print(t.get_ascii(attributes=['name'], show_internal=True)))

    G = translate_to_networkx_graph(t)
    pass
    # for i, pruning_head_node in enumerate(t.iter_descendants("levelorder")):
    #     if not pruning_head_node.up.is_root(): # if this is not one of the two direct child nodes of the root
    #         pruning_edge = Edge(node_a=pruning_head_node.name, node_b=pruning_head_node.up.name)
    #     for j, regrafting_head_node in enumerate(t.iter_descendants("levelorder")):
    #         if not regrafting_head_node.up.is_root():
    #             regrafting_edge = Edge(node_a=regrafting_head_node.name, node_b=regrafting_head_node.up.name)
    #             if not ((pruning_edge.node_a == regrafting_edge.node_a) or (pruning_edge.node_b == regrafting_edge.node_b) or (
    #                     pruning_edge.node_b == regrafting_edge.node_a) or (pruning_edge.node_a == regrafting_edge.node_b))



if __name__ == "__main__":
    main()
