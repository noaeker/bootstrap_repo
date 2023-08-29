from ete3 import Tree

def root_tree(t):
    """
    :param t: an ETE3 tree
    :return: a rooted tree (by t's first child)
    """
    rooted_tree = t.copy()
    ancestor = rooted_tree.get_children()[0]
    rooted_tree.set_outgroup(ancestor)
    return rooted_tree


def name_internal_nodes(tree, st=""):
    """
    :param tree: ete3 Tree
    :param st: the name for the internal node (will add to the serial number), default - empty string
    :return: the function sets names to all internal nodes and returns a list of all new names
    """
    i = 0
    tree.name = "root"
    for node in tree.iter_descendants("preorder"):
        if not node.is_leaf():
            node.name = st+str(i)
            i += 1
    internal_nodes = [st+str(k) for k in range(i)]
    return internal_nodes


def swap_by_edge(edge, file_path, show):
    """
    :param edge: an internal edge - represented by a list of two nodes
    :param file_path: path to the tree's file
    :param show: boolean parameter - True if user wants to see the trees, otherwise False
    :return: a list of two NNI neighbors, by "edge" input
    """
    second_node_children = edge[1].get_children()

    # create first copy and NNI neighbor
    new_tree1 = Tree(file_path, format=1)
    new_tree1 = root_tree(new_tree1)
    name_internal_nodes(new_tree1)
    make_permutation(new_tree1, edge, second_node_children, 0, show)

    # create second copy and NNI neighbor
    new_tree2 = Tree(file_path, format=1)
    new_tree2 = root_tree(new_tree2)
    name_internal_nodes(new_tree2)

    make_permutation(new_tree2, edge, second_node_children, 1, show)
    return [new_tree1, new_tree2]


def make_permutation(new_tree, edge, second_node_children, idx, show):
    """
    :param new_tree: a copy of the original tree
    :param edge: an internal edge
    :param second_node_children: the second node (closer to leaf) children
    :param idx: 0 or 1 according to first or second NNI neighbor
    :param show: boolean parameter - True if user wants to see the trees, otherwise False
    :return: the function swaps between two children to create a NNI neighbor
    note that I assume that the first node in edge is the upper one (follows the implementation)
    """
    for node in edge[0].get_children():
        if node not in edge:
            node1 = node

    node2 = second_node_children[idx]

    parent1 = new_tree & edge[0].name
    parent2 = new_tree & edge[1].name

    parent1.remove_child(new_tree & node1.name)
    parent2.remove_child(new_tree & node2.name)
    parent1.add_child(node2)
    parent2.add_child(node1)
    if show:
        print(f"\nneighbors by edge=({edge[0].name},{edge[1].name}), nodes swapped=({node1.name},{node2.name})\n",
          new_tree.get_ascii(show_internal=True))


def NNI_neighbors(file_path, show=False):
    """
    :param file_path: path to the tree's file
    :param show: optional boolean parameter (default - False) - True if user wants to see the trees.
    :return: rooted_tree: an ETE3 object represents the tree, rooted by root's first child
             neighbors: a list of all rooted_tree's NNI neighbors
    """
    tree = Tree(file_path, format=1)
    if show:
        print("\noriginal tree:\n", tree.get_ascii(show_internal=True))

    rooted_tree = root_tree(tree)
    internal_n = name_internal_nodes(rooted_tree)
    if show:
        print("\nrooted tree:\n", rooted_tree.get_ascii(show_internal=True))

    node_child_dict = {rooted_tree: rooted_tree.get_children()}
    edges = []

    for node in rooted_tree.iter_descendants("preorder"):
        if not node.is_leaf():
            node_child_dict[node] = node.get_children()
            for i in range(len(node_child_dict[node])):  # add all edges from a parent to its children
                node_child_dict[node][i] = node_child_dict[node][i]
                edges.append([node, node_child_dict[node][i]])
    edges.append([node_child_dict[rooted_tree][0], node_child_dict[rooted_tree][1]])
    in_edges= [edges[i] for i in range(len(edges)) if edges[i][0].name in internal_n and edges[i][1].name in internal_n]

    neighbors = []
    for edge in in_edges:
        neighbors.extend(swap_by_edge(edge, file_path, show))

    return rooted_tree, neighbors


if __name__ == '__main__':

    tree, all_neighbors = NNI_neighbors("unrooted.newick", show=True)
    #print(f"\nnumber of NNI neighbors: {len(all_neighbors)}\n"
    #      f"number of nodes: {len(tree.get_leaves())}\n"
    #      f"does fit formula? {len(all_neighbors) == 2*(len(tree.get_leaves()) - 3)}")