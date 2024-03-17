def nd_to_index(Graph):
    nodes_idx = {node: i for i, node in enumerate(Graph.nodes)}
    return nodes_idx