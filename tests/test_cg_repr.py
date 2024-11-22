import unittest
import numpy as np
from compgraph.cg_repr import *
import quimb as qu
import networkx as nx
from compgraph.useful import config_to_state, node_to_index, graph_tuple_list_to_configs_list, config_list_to_state_list, neel_state, state_from_config_amplitudes
from compgraph.cg_repr import graph_tuple_to_config_hamiltonian_product_update
from compgraph.tensor_wave_functions import sparse_tensor_exp_energy, create_sparsetensor_from_configs_amplitudes
import itertools
import tensorflow as tf
class TestCgRepr(unittest.TestCase):
    def setUp(self) -> None:
        self.config= np.array([-1, -1, 1])
        return super().setUp()
    

    def test_apply_raising_operator(self):
        site = 0
        expected_config = np.array([1, -1, 1])
        new_config = apply_raising_operator(self.config, site)
        self.assertTrue(np.isclose(new_config, expected_config).all())
        self.assertFalse(np.isclose(self.config, new_config).all())
        
        # Test that applying the operator to a spin-up state returns None
        new_config = apply_raising_operator(expected_config, site)
        self.assertIsNone(new_config)

    def test_apply_lowering_operator(self):
        site =2
        expected_config = np.array([-1, -1, -1])
        new_config = apply_lowering_operator(self.config, site)

        if new_config is not None:

            self.assertTrue(np.isclose(new_config, expected_config).all())
            self.assertFalse(np.isclose(self.config, new_config).all())
        # Test that applying the operator to a spin-down state returns None
        new_config = apply_lowering_operator(new_config, site)
        self.assertIsNone(new_config)


    def test_are_configs_identical(self):
        config1 = np.array([1, -1, 1])
        config2 = np.array([1, -1, 1])
        self.assertTrue(are_configs_identical(config1, config2))
        config3 = np.array([-1, -1, 1])
        self.assertFalse(are_configs_identical(config1, config3))

    def test_configs_differ_by_two_sites(self):
        config1 = np.array([1, -1, 1])
        config2 = np.array([-1, 1, 1])
        self.assertTrue(configs_differ_by_two_sites(config1, config2))
        config3 = np.array([-1, -1, 1])
        self.assertFalse(configs_differ_by_two_sites(config1, config3))
        config4 = np.array([1, -1, -1])
        self.assertFalse(configs_differ_by_two_sites(config1, config4))


    def test_config_hamiltonian_product(self):
        n= 2
        m=2
        lattice_size = (n,m)
        G = nx.grid_2d_graph(*lattice_size, periodic=True)
        mapping = node_to_index(G)
        G = nx.relabel_nodes(G, mapping)
        sub_lattice_encoding=neel_state(G) 
        full_basis_configs = np.array([[int(x) for x in format(i, f'0{n*m}b')] for i in range(2**(n*m))]) * 2 - 1

        for configuration in full_basis_configs:
            #config =np.array([[1, -1, -1, -1]])
            #check if digraph gives same as the function implemented

            #lattice=nx.DiGraph()
            #lattice.add_nodes_from(range(4))
            #lattice.add_edges_from([[0,1],[1,2],[2,3],[3,0],[1,0],[2,1],[3,2],[0,3]])

            config=np.array([configuration])
            Hamiltonian= qu.ham_heis_2D(n, m, j=1.0, bz=0, cyclic=True, parallel=False, ownership=None)
            psi= config_to_state(config[0])
            mat_vec= Hamiltonian @ psi

            site=create_graph_tuples(config, G, sub_lattice_encoding)
            configurations_tuples, coefficients = graph_tuple_to_config_hamiltonian_product_update(site[0], G, sub_lattice_encoding)
            configurations_list= graph_tuple_list_to_configs_list(configurations_tuples)
            states=(config_list_to_state_list(configurations_list))
            scaled_states = [coeff * state for coeff, state in zip(coefficients, states)]
            vec_from_states = sum(scaled_states)

            #print(coefficients, vec_from_states,mat_vec)
            self.assertTrue(np.allclose(mat_vec,vec_from_states))
    
    def test_config_to_state(self):
        config=np.array([1])
        generated1=config_to_state(config)
        self.assertTrue(np.allclose(generated1,qu.down()))
        config=np.array([1,-1,1])
        generated1=config_to_state(config)
        self.assertTrue(np.allclose(generated1,qu.down()&qu.up()&qu.down()))
    
    
    def test_sparse_tensor_exp_energy(self):
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
            amplitudes = np.random.rand(2**(n*m)) + 1j * np.random.rand(2**(n*m))  # Random complex amplitudes
            # Compute energy using sparse_tensor_exp_energy
            J2 = 1.0  # Example J2 value
            sparse_tensor=create_sparsetensor_from_configs_amplitudes(full_basis_configs, amplitudes, num_sites)
            computed_energy = sparse_tensor_exp_energy(sparse_tensor, G, 0)
            #print(amplitudes, type(amplitudes),amplitudes.shape)

            psi_total=state_from_config_amplitudes(full_basis_configs, amplitudes)
            #print("psi total", psi_total)
            Hamiltonian = qu.ham_heis_2D(n, m, j=1.0, bz=0, cyclic=True)
            expected_energy = psi_total.H @ Hamiltonian @ psi_total
            norm= qu.norm(psi_total)
            print(norm, tf.norm(sparse_tensor.values), np.sqrt(psi_total.H@psi_total))
            # Check if energies are close
            self.assertTrue(np.allclose(computed_energy, expected_energy/norm**2))
            
    

    #TODO Check that time_evoluted function is doing what it should do (similar syntax as test config_ham, but this time
    #we need to check that it has indeed the form (1- bH)|psi>; against configurations on lattice 2x2 and 3x3
    #Build a model as a test function that takes a graph_tuple and gives back a complex coefficient for each different graph_tuple configuration
    def test_config_hamiltonian_product(self):
        n= 2
        m=2
        lattice_size = (n,m)
        G = nx.grid_2d_graph(*lattice_size, periodic=True)
        mapping = node_to_index(G)
        G = nx.relabel_nodes(G, mapping)
        sub_lattice_encoding=neel_state(G) 
        full_basis_configs = np.array([[int(x) for x in format(i, f'0{n*m}b')] for i in range(2**(n*m))]) * 2 - 1

        for configuration in full_basis_configs:
            #config =np.array([[1, -1, -1, -1]])
            #check if digraph gives same as the function implemented

            #lattice=nx.DiGraph()
            #lattice.add_nodes_from(range(4))
            #lattice.add_edges_from([[0,1],[1,2],[2,3],[3,0],[1,0],[2,1],[3,2],[0,3]])

            config=np.array([configuration])
            Hamiltonian= qu.ham_heis_2D(n, m, j=1.0, bz=0, cyclic=True, parallel=False, ownership=None)
            psi= config_to_state(config[0])
            mat_vec= Hamiltonian @ psi

            site=create_graph_tuples(config, G, sub_lattice_encoding)
            configurations_tuples, coefficients = graph_tuple_to_config_hamiltonian_product_update(site[0], G)
            configurations_list= graph_tuple_list_to_configs_list(configurations_tuples)
            states=(config_list_to_state_list(configurations_list))
            scaled_states = [coeff * state for coeff, state in zip(coefficients, states)]
            vec_from_states = sum(scaled_states)

            #print(coefficients, vec_from_states,mat_vec)
            self.assertTrue(np.allclose(mat_vec,vec_from_states))
    
if __name__ == '__main__':
    unittest.main()
    