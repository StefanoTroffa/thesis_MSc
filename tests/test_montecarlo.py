import unittest
import numpy as np
import quimb as qu
import networkx as nx
from compgraph.cg_repr import *
from compgraph.useful import create_graph_tuples, node_to_index, neel_state, state_from_config_amplitudes
from compgraph.gnn_src_code import GNN_double_output
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, sparse_tensor_exp_energy, create_sparsetensor_from_configs_amplitudes, time_evoluted_wave_function_on_batch, montecarlo_logloss_overlap_time_evoluted, calculate_sparse_overlap, quimb_vec_to_sparse
import itertools
import tensorflow as tf

    
class TestMonteCarlofunctions(unittest.TestCase):

    def test_montecarlo_logloss_overlap_time_evoluted(self):
        """
        Tests the Monte Carlo log loss overlap time evolved function. This function:
        - Initializes a model.
        - Applies the variational wave function on a batch of configurations.
        - Transforms TensorFlow sparse tensor outputs into quimb state representations.
        - Computes the Hamiltonian and evolves the state using quimb.
        - Calculates the overlap of the initial state with the time-evolved state.
        - Normalizes and logs the overlap to compute the log loss.
        - Compares the result with the expected log loss from the function under test.

        """
        n, m = 2, 2
        num_sites = n * m
        lattice_size = (n, m)
        G = nx.grid_2d_graph(*lattice_size, periodic=True)
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        sublattice_enc = neel_state(G)
        graph_hamiltonian = qu.ham_heis_2D(n, m, j=1.0, bz=0, cyclic=True, sparse=True)
        beta = 0.05
        model_var = GNN_double_output()

        # Simulate multiple tests
        n_tests = num_sites * 2
        for i in range(n_tests):

            # Generate random configurations
            full_basis_configs = np.array([[int(x) for x in format(i, f'0{n*m}b')] for i in range(2**(n*m))]) * 2 - 1

            # Convert configurations to graph tuples
            graph_tuples_var =create_graph_tuples(full_basis_configs,G, sublattice_enc)
            graph_tuples_te = graph_tuples_var

            # Calculate coefficients
            sparse_coefficients_var = variational_wave_function_on_batch(model_var, graph_tuples_var)
            sparse_coefficients_te = time_evoluted_wave_function_on_batch(model_var, beta, graph_tuples_te, G, sublattice_enc)

            # Calculate the monte carlo log loss overlap
            computed_logloss = montecarlo_logloss_overlap_time_evoluted(sparse_coefficients_te, graph_tuples_te, model_var, model_var, graph_tuples_var, beta, G, sublattice_enc)
            #print(computed_logloss)

            amplitudes =np.array(sparse_coefficients_var.values)

            # Compute expected states and overlap
            psi = state_from_config_amplitudes(full_basis_configs, amplitudes)
            psi_te=state_from_config_amplitudes(full_basis_configs, np.array(sparse_coefficients_te.values))
            #psi_te2 = psi - beta * graph_hamiltonian@psi
            #print(psi_te-psi_te2)

            #print("Confronting psi and the state version of it \n", psi, 'vs', sparse_coefficients_var)
            #print("Confronting psi t.e. and the state version of it \n", psi_te, 'vs', sparse_coefficients_te)
            #print(psi_te1, 'vs', psi_te)
            overlap = psi.H@psi_te * psi_te.H@psi
            normalized_overlap = tf.math.sqrt(overlap) / (qu.norm(psi) * qu.norm(psi_te))
            #print('norm psi', qu.norm(psi),tf.norm(sparse_coefficients_var.values), psi.H@psi)
            #print("Normalized overlap according to quimb", overlap, 'left part and tight part',psi.H@psi_te, psi_te.H@psi)
            expected_logloss = -tf.math.log(((normalized_overlap)))

            # Assert the computed and expected logloss are close
            self.assertTrue(np.isclose(computed_logloss, expected_logloss), f"Failed on test {i}: Computed {computed_logloss}, Expected {expected_logloss}")
    
    def test_calculate_sparse_overlap(self):
        n, m = 2, 2
        num_sites = n * m
        lattice_size = (n, m)
        G = nx.grid_2d_graph(*lattice_size, periodic=True)
        mapping = node_to_index(G)
        G = nx.relabel_nodes(G, mapping)
        n_tests= num_sites*2
        for i in range(n_tests):
            # Generate random configurations and amplitudes
            full_basis_configs = np.array([[int(x) for x in format(i, f'0{n*m}b')] for i in range(2**(n*m))]) * 2 - 1
            num_configs = len(full_basis_configs)  # Define how many random states you want to test

            amplitudes_right = np.random.rand(num_configs) + 1j * np.random.rand(num_configs)  # Random complex amplitudes
            amplitudes_left= np.random.rand(num_configs) + 1j * np.random.rand(num_configs)  # Random complex amplitudes
            # Compute energy using sparse_tensor_exp_energy
            sparse_tensor_right=create_sparsetensor_from_configs_amplitudes(full_basis_configs, amplitudes_right, num_sites)
            sparse_tensor_left=create_sparsetensor_from_configs_amplitudes(full_basis_configs, amplitudes_left, num_sites)

            computed_energy_right = sparse_tensor_exp_energy(sparse_tensor_right, G, 0)
            computed_energy_left = sparse_tensor_exp_energy(sparse_tensor_left, G, 0)


            psi_right=state_from_config_amplitudes(full_basis_configs, amplitudes_right)
            psi_left = state_from_config_amplitudes(full_basis_configs, amplitudes_left)
            print(psi_right, sparse_tensor_right.values)

            Hamiltonian = qu.ham_heis_2D(n, m, j=1.0, bz=0, cyclic=True)
            expected_energy_right = psi_right.H @ Hamiltonian @ psi_right
            expected_energy_left = psi_left.H @ Hamiltonian @ psi_left
            overlap_left_right= calculate_sparse_overlap(sparse_tensor_left,sparse_tensor_right )
            ovl= psi_left.H@psi_right

            print(overlap_left_right, ovl)
            # Check if energies are close
            self.assertTrue(np.allclose(computed_energy_right,expected_energy_right/qu.norm(psi_right)**2))
            self.assertTrue(np.allclose(computed_energy_left,expected_energy_left/qu.norm(psi_left)**2))
            self.assertTrue(np.allclose(overlap_left_right, ovl))
    
    def test_quimb_to_vec(self):
        n, m = 2, 2
        num_sites = n * m
        lattice_size = (n, m)
        G = nx.grid_2d_graph(*lattice_size, periodic=True)
        mapping = node_to_index(G)
        G = nx.relabel_nodes(G, mapping)
        n_tests= num_sites*2
        for i in range(n_tests):
            # Generate random configurations and amplitudes
            full_basis_configs = np.array([[int(x) for x in format(i, f'0{n*m}b')] for i in range(2**(n*m))]) * 2 - 1
            print(full_basis_configs)
            
            num_configs = len(full_basis_configs)  # Define how many random states you want to test

            amplitudes= np.random.rand(num_configs) + 1j * np.random.rand(num_configs)  # Random complex amplitudes
            # Compute energy using sparse_tensor_exp_energy
            sparse_tensor=create_sparsetensor_from_configs_amplitudes(full_basis_configs, amplitudes, num_sites)

            psi = state_from_config_amplitudes(full_basis_configs, amplitudes)
            sparse_from_qu= quimb_vec_to_sparse(psi,full_basis_configs, len(G.nodes))
            #print(overlap_left_right, ovl)
            # Check if energies are close
            print(psi, sparse_tensor.values,amplitudes)
            self.assertTrue(np.allclose(sparse_tensor.values,sparse_from_qu.values))



if __name__ == '__main__':
    unittest.main()