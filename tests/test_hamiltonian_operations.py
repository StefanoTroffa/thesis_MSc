import unittest
import numpy as np
from compgraph.cg_repr import *
import quimb as qu
import networkx as nx
from compgraph.useful import config_to_state, node_to_index, graph_tuple_list_to_configs_list, config_list_to_state_list, state_from_config_amplitudes
from compgraph.cg_repr import graph_tuple_to_config_hamiltonian_product_update
from compgraph.tensor_wave_functions import sparse_tensor_exp_energy, create_sparsetensor_from_configs_amplitudes
import tensorflow as tf
from compgraph.tensorflow_version.hamiltonian_operations import config_hamiltonian_product_jit_o3, config_hamiltonian_product_xla_improved
from simulation.initializer import neel_encoding_from_graph
from compgraph.useful import create_graph_tuples, node_to_index, state_from_config_amplitudes, config_to_state
from compgraph.models import GNN_double_output
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, sparse_tensor_exp_energy, create_sparsetensor_from_configs_amplitudes, time_evoluted_wave_function_on_batch, montecarlo_logloss_overlap_time_evoluted, calculate_sparse_overlap, quimb_vec_to_sparse
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, sparse_tensor_exp_energy, create_sparsetensor_from_configs_amplitudes, time_evoluted_wave_function_on_batch, montecarlo_logloss_overlap_time_evoluted, calculate_sparse_overlap, quimb_vec_to_sparse
from simulation.initializer import create_graph_from_ham, neel_encoding_from_graph
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure
from compgraph.tensorflow_version.hamiltonian_operations import stochastic_gradients_tfv3, stochastic_energy_tf

import itertools
from compgraph.monte_carlo import MCMCSampler,compute_phi_terms, stochastic_energy
from compgraph.models import GNN_double_output
from compgraph.useful import (
    create_graph_tuples,
    config_to_state,
    state_from_config_amplitudes,
    create_amplitude_frequencies_from_graph_tuples,
    generate_graph_tuples_configs,
    sparse_list_to_configs,
)
from simulation.initializer import (
    create_graph_from_ham,
    initialize_NQS_model_fromhyperparams,
    initialize_graph_tuples,
    neel_encoding_from_graph
)


class OverlapFunctions(unittest.TestCase):
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
    
