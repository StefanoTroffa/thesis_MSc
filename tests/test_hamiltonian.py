from compgraph import cg_repr as cg

import unittest
import numpy as np
import netket as nk
import networkx as nx
from compgraph.sparse_ham import construct_sparse_hamiltonian  # Adjust the import based on your project structure


class test_HamiltonianWithNetKet(unittest.TestCase):
    def test_two_node_graph_netket(self):
        # Setup: Create a simple graph with NetKet
        graph = nk.graph.Graph(edges=[[0, 1]])
        hilbert_space = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

        # Define the Heisenberg Hamiltonian with NetKet
        hamiltonian_nk = nk.operator.Heisenberg(hilbert=hilbert_space, graph=graph, J=0.25, sign_rule=True)

        # Convert NetKet Hamiltonian to Scipy sparse matrix for comparison
        H_nk_sparse = hamiltonian_nk.to_sparse()

        # Setup: Create the same graph with NetworkX
        G = nx.Graph()
        G.add_edge(0, 1)
        print(G, graph)
        # Exercise: Generate Hamiltonian using your function
        H_test = construct_sparse_hamiltonian(G, 0)

        # Verify: Ensure the matrices are similar enough
        # Since these are sparse matrices, we can convert them to dense form for comparison
        H_nk_dense = H_nk_sparse.toarray()
        H_test_dense = H_test.toarray()

        self.assertTrue(np.allclose(H_nk_dense, H_test_dense, atol=1e-7))

if __name__ == '__main__':
    unittest.main()