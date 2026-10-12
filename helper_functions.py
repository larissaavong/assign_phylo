"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    
    # self.tree = Node()
    n = len(labels)
    
    leaf_nodes = []
    for label in labels:
        leaf_nodes.append(Node(label, None, 0.0, None, 0.0))

    dist = np.array(distances, dtype=float)
    node_counter = 1

    while n > 2:
        # calculate sum of each row 
        sum_of_distances = np.sum(dist, axis=1)

        # calculate adjusted distance matrix (Q)
        Q_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    Q_matrix[i][j] = np.inf
                else:
                    Q_matrix[i][j] = (n - 2) * dist[i, j] - sum_of_distances[i] - sum_of_distances[j]
                    Q_matrix[j][i] = Q_matrix[i][j]

        # find distinct nodes i != j for which Q(i, j) is minimal
        flat_index = np.argmin(Q_matrix)
        i, j = np.unravel_index(flat_index, Q_matrix.shape)

        if i > j:
            i, j = j, i

        delta_i = 0.5 * dist[i, j] + 1/(2 * (n - 2)) * (sum_of_distances[i] - sum_of_distances[j])
        delta_j = dist[i, j] - delta_i

        left_child = leaf_nodes[i]
        right_child = leaf_nodes[j]

        node_name = f"node{node_counter}"
        new_node = Node(node_name, left_child, delta_i, right_child, delta_j)
        node_counter += 1

        # remove left and right child from leaf_node list + append new_node to leaf 
        leaf_nodes.remove(left_child)
        leaf_nodes.remove(right_child)
        leaf_nodes.append(new_node)

        # get list of all indices that are not being currently merged 
        remaining_indices = []
        for k in range(n):
            if k != i and k != j:
                remaining_indices.append(k)

        # distance between every node and merged node
        new_distance = []
        for k in remaining_indices:
            distance = 0.5 * (dist[i, k] + dist[j, k] - dist[i, j])
            new_distance.append(distance)

        reduced_distances = dist[np.ix_(remaining_indices, remaining_indices)]
        new_distance = np.array(new_distance)

        new_n = n - 1
        updated_dist = np.zeros((new_n, new_n))
        updated_dist[:-1, :-1] = reduced_distances

        updated_dist[:-1, -1] = new_distance
        updated_dist[-1, :-1] = new_distance
        updated_dist[-1, -1] = 0.0

        dist = updated_dist
        n -= 1

    final_dist = dist[0, 1] / 2
    root = Node("root", leaf_nodes[0], final_dist, leaf_nodes[1], final_dist)

    return root


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    raise NotImplementedError()


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()