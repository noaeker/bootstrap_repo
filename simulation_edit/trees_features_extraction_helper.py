
from side_code.basic_trees_manipulation import *
import numpy as np

def get_neighboring_nodes(tree_ete, node_name):
    tree_ete_cp = tree_ete.copy()
    node_pair1 = [node_c for node_c in (tree_ete_cp & node_name).children]
    if (tree_ete_cp & node_name).up.is_root():
        node_pair2 = [node for node in tree_ete.get_tree_root().children if node.name != node_name]

    else:
        sister_node = [node for node in (tree_ete_cp & node_name).up.children if node.name != node_name]
        parent_node = [(tree_ete_cp & node_name).up]
        node_pair2 = sister_node + parent_node
    # sister_children_nodes = \
    #    ([node_c for node_c in (tree_ete_cp & node_name).up.children if node_c.name != node_name][0]).children
    return node_pair1, node_pair2

def get_branch_lengths(tree):
    branch_lengths = []
    for node in tree.iter_descendants():
        # Do some analysis on node
        branch_lengths.append(node.dist)
    return branch_lengths


def get_tree_divergence(tree):
    branch_lengths = get_branch_lengths(tree)
    return np.sum(branch_lengths)


def mad_tree_parameter(tree_path, program_path = None):
    if program_path is None:
        program_path = MAD_COMMAND_PREFIX
    mad_command = "{mad_exe_path} -t -s {tree_path}".format(mad_exe_path=program_path,
                                                            tree_path=tree_path)
    execute_command_and_write_to_log(mad_command)
    mad_log_path = tree_path + ".rooted"
    mad = extract_mad_file_statistic(mad_log_path)
    os.remove(mad_log_path)
    return mad

def get_booster_tree(mle_tree_path, comparison_tree, out_path ="booster.nw", tbe = False, program_path = None):
    algo = 'tbe' if tbe else 'fbp'
    if program_path is None:
        program_path =BOOSTER_EXE
    cmd = f"{program_path} -a {algo} -i {mle_tree_path} -b {comparison_tree} -@ 1 -o {out_path}"
    execute_command_and_write_to_log(cmd)
    with open(out_path) as B:
        bootster_tree = B.read()
    try:
        booster_tree_ete = Tree(newick=bootster_tree, format=0)
    except:
        pass
    add_internal_names(booster_tree_ete)
    return booster_tree_ete




def get_bootstrap_support(garbage_dir, mle_path, comparison_trees_path, program_path = None):
    tbe_comp_tree = get_booster_tree(mle_path, comparison_trees_path,
                                     out_path=os.path.join(garbage_dir, "tbe_booster_pars.nw"), program_path=program_path,tbe= True)
    fbp_comp_tree = get_booster_tree(mle_path, comparison_trees_path,
                                     out_path=os.path.join(garbage_dir, "fbp_booster_pars.nw"),program_path=program_path, tbe = False)
    return tbe_comp_tree, fbp_comp_tree


def generate_tree(node_pair1, node_pair2):
    sub_tree1 = Tree()
    sub_tree1.add_child(node_pair1[0])
    sub_tree1.add_child(node_pair1[1])
    sub_tree2 = Tree()
    sub_tree2.add_child(node_pair2[0])
    sub_tree2.add_child(node_pair2[1])
    final_tree = Tree()
    final_tree.add_child(sub_tree1)
    final_tree.add_child(sub_tree2)
    return final_tree


def get_nni_neighbors(tree_path, node_name):
    tree = Tree(tree_path, format=1)
    tree.set_outgroup(tree & node_name)
    node_sister = [node for node in (tree & node_name).up.children if node.name != node_name][0]
    cousins = node_sister.children
    kids = (tree & node_name).children
    pruned_first_child = (kids[0]).detach()
    pruned_second_child = (kids[0]).detach()
    pruned_first_cousin = (cousins[0]).detach()
    pruned_second_cousin = (cousins[0]).detach()
    first_final_tree = generate_tree((pruned_first_child, pruned_first_cousin),
                                     (pruned_second_child, pruned_second_cousin))
    second_final_tree = generate_tree((pruned_first_child, pruned_second_cousin),
                                      (pruned_second_child, pruned_first_cousin))
    neighbors = (first_final_tree, second_final_tree)
    return neighbors