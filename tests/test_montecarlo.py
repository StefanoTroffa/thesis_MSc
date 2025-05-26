import unittest
import numpy as np
import quimb as qu
import networkx as nx
import tensorflow as tf
from compgraph.cg_repr import *

from compgraph.useful import create_graph_tuples, node_to_index, state_from_config_amplitudes, config_to_state
from compgraph.models import GNN_double_output
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, sparse_tensor_exp_energy, create_sparsetensor_from_configs_amplitudes, time_evoluted_wave_function_on_batch, montecarlo_logloss_overlap_time_evoluted, calculate_sparse_overlap, quimb_vec_to_sparse
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, sparse_tensor_exp_energy, create_sparsetensor_from_configs_amplitudes, time_evoluted_wave_function_on_batch, montecarlo_logloss_overlap_time_evoluted, calculate_sparse_overlap, quimb_vec_to_sparse
from simulation.initializer import create_graph_from_ham, neel_encoding_from_graph
import itertools
from compgraph.monte_carlo import MCMCSampler
from compgraph.models import GNN_double_output
from compgraph.monte_carlo import MCMCSampler, stochastic_energy
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
)
    
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
        sublattice_enc = neel_encoding_from_graph(G)
        graph_hamiltonian = qu.ham_heis_2D(n, m, j=1.0, bz=0, cyclic=True, sparse=True)

        # Simulate multiple tests
        n_tests = num_sites * 2
        for i in range(n_tests):
            model_var = GNN_double_output(32,16)
            beta = 0.05*i
            model_te=GNN_double_output(32,16)
            # Generate random configurations
            full_basis_configs = np.array([[int(x) for x in format(i, f'0{n*m}b')] for i in range(2**(n*m))]) * 2 - 1

            # Convert configurations to graph tuples
            graph_tuples_var =create_graph_tuples(full_basis_configs,G, sublattice_enc)
            graph_tuples_te = graph_tuples_var

            # Calculate coefficients
            sparse_coefficients_var = variational_wave_function_on_batch(model_var, graph_tuples_var)
            sparse_coefficients_te = time_evoluted_wave_function_on_batch(model_te, beta, graph_tuples_te, G)

            # Calculate the monte carlo log loss overlap
            computed_logloss = montecarlo_logloss_overlap_time_evoluted(sparse_coefficients_te, graph_tuples_te, model_var, model_te, graph_tuples_var, beta, G)
            #print(computed_logloss)

            amplitudes =np.array(sparse_coefficients_var.values)

            # Compute expected states and overlap
            psi = state_from_config_amplitudes(full_basis_configs, amplitudes)
            psi_te=state_from_config_amplitudes(full_basis_configs, np.array(sparse_coefficients_te.values))
            #psi_te2 = psi - beta * graph_hamiltonian@psi
            # print(psi-psi_te)



            #print("Confronting psi and the state version of it \n", psi, 'vs', sparse_coefficients_var)
            #print("Confronting psi t.e. and the state version of it \n", psi_te, 'vs', sparse_coefficients_te)
            #print(psi_te1, 'vs', psi_te)
            overlap = psi.H@psi_te * psi_te.H@psi
            normalized_overlap = tf.math.sqrt(overlap) / (qu.norm(psi) * qu.norm(psi_te))
            #print('norm psi', qu.norm(psi),tf.norm(sparse_coefficients_var.values), psi.H@psi)
            #print("Normalized overlap according to quimb", overlap, 'left part and tight part',psi.H@psi_te, psi_te.H@psi)
            expected_logloss = -tf.math.log(((normalized_overlap)))
            print(computed_logloss, expected_logloss)
            # Assert the computed and expected logloss are close
            self.assertTrue(np.isclose(computed_logloss, expected_logloss), f"Failed on test {i}: Computed {computed_logloss}, Expected {expected_logloss}")
    def test_FD_montecarlo_logloss_overlap_time_evoluted(self):
        """
        Computes finite difference gradients and compares them with TensorFlow's automatic differentiation.
        """
        n, m = 2, 2
        num_sites = n * m
        lattice_size = (n, m)
        G = nx.grid_2d_graph(*lattice_size, periodic=True)
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        sublattice_enc = neel_encoding_from_graph(G)
        graph_hamiltonian = qu.ham_heis_2D(n, m, j=1.0, bz=0, cyclic=True, sparse=True)
        beta = 0.05
        model_var = GNN_double_output(32, 16)
        model_te = GNN_double_output(32, 16)
        epsilon = 1e-5  # Small perturbation for finite difference

        # Generate random configurations
        full_basis_configs = np.array([[int(x) for x in format(i, f'0{n*m}b')] for i in range(2**(n*m))]) * 2 - 1

        # Convert configurations to graph tuples
        graph_tuples_var = create_graph_tuples(full_basis_configs, G, sublattice_enc)
        graph_tuples_te = graph_tuples_var

        # Calculate coefficients
        sparse_coefficients_var = variational_wave_function_on_batch(model_var, graph_tuples_var)
        sparse_coefficients_te = time_evoluted_wave_function_on_batch(model_te, beta, graph_tuples_te, G)

        # Calculate the monte carlo log loss overlap
        with tf.GradientTape() as tape:
            computed_logloss = montecarlo_logloss_overlap_time_evoluted(sparse_coefficients_te, graph_tuples_te, model_var, model_te, graph_tuples_var, beta, G)
        # Compute gradients with respect to the model's trainable variables
        analytical_gradients = tape.gradient(computed_logloss, model_var.trainable_variables)

        # Now, compute finite difference gradients
        finite_diff_gradients = []
        for var in model_var.trainable_variables:
            var_shape = var.shape
            var_grad = np.zeros(var_shape)
            var_np = var.numpy()

            # Iterate over each element in the variable
            for index, _ in np.ndenumerate(var_np):
                # Perturb variable by epsilon
                var_plus = var_np.copy()
                var_minus = var_np.copy()
                var_plus[index] += epsilon
                var_minus[index] -= epsilon

                # Assign the perturbed values to the variable and compute the loss
                var.assign(var_plus)
                loss_plus = montecarlo_logloss_overlap_time_evoluted(
                    sparse_coefficients_te,
                    graph_tuples_te, model_var, model_var, graph_tuples_var, beta, G
                )

                var.assign(var_minus)
                loss_minus = montecarlo_logloss_overlap_time_evoluted(
                    sparse_coefficients_te,
                    graph_tuples_te, model_var, model_te, graph_tuples_var, beta, G
                )

                # Compute finite difference gradient
                var_grad[index] = (loss_plus - loss_minus) / (2 * epsilon)

            # Append the computed gradient
            finite_diff_gradients.append(var_grad)

            # Reset the variable to its original value
            var.assign(var_np)

        # Compare the finite difference gradients with the analytical gradients
        for i, (fd_grad, an_grad) in enumerate(zip(finite_diff_gradients, analytical_gradients)):
            self.assertTrue(np.allclose(fd_grad, an_grad, atol=1e-5),
                            f"Finite difference gradient and analytical gradient do not match for variable {i}.")
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
    #def test_proposed_tuples(self):
    def test_ite_in_mcmc_sampler(self):
        """
        Test the MCMC Sampler for consistency in the wavefunction calculation of time evolution.
        """
        hyperparams = {
            'graph_params': {
                'graphType': '2dsquare',
                'n': 2,
                'm': 2,
                'sublattice': 'Neel'
            },
            'sim_params': {
                'beta': 0.05,
                'batch_size': 4,
                'learning_rate': 7e-5,
                'n_batch': 1,
            },
            'ansatz': 'GNN2simple',
            'ansatz_params': {
                'K_layer': 1,
                'hidden_size': 8,
                'output_emb_size': 4
            }
        }

        # Initialize graph and sampler
        graph, subl = create_graph_from_ham(
            hyperparams['graph_params']['graphType'],
            (hyperparams['graph_params']['n'], hyperparams['graph_params']['m']),
            hyperparams['graph_params']['sublattice']
        )

        full_basis_configs = np.array([[int(x) for x in format(i, f'0{len(graph.nodes)}b')] for i in range(2**(len(graph.nodes)))]) * 2 - 1
        full_basis_gt = create_graph_tuples(full_basis_configs, graph, subl)
        model = GNN_double_output(128, 32)
        beta = 0.12
        sampler_te = MCMCSampler(model, full_basis_gt[0], beta, graph)

        wave_function_te_on_full = [sampler_te.evaluate_model(gt) for gt in full_basis_gt]
        amplitudes = np.array(wave_function_te_on_full)

        psi_full_state_vec = state_from_config_amplitudes(full_basis_configs, amplitudes)
        Hamiltonian = qu.ham_heis_2D(hyperparams['graph_params']['n'], hyperparams['graph_params']['m'], j=1.0, bz=0, cyclic=True, parallel=False, ownership=None)

        for index, gt in enumerate(full_basis_gt):
            total_amplitude = sampler_te.time_evoluted_config_amplitude(gt)
            psi_amplitude = sampler_te.evaluate_model(gt)

            psi_ket = config_to_state(full_basis_configs[index])
            H_psi_beta = beta * (Hamiltonian @ psi_full_state_vec)
            psi_te_quimb = psi_amplitude * psi_ket - H_psi_beta

            # Assert the computed amplitude from time evolution matches expected value
            self.assertTrue(np.allclose(psi_te_quimb[index], total_amplitude),
                            f"Mismatch at index {index}: Computed {total_amplitude}, Expected {psi_te_quimb[index]}")

class TestMCMCEnergyTrajectory(unittest.TestCase):
    def setUp(self):
        # Allow for easy personalization of the lattice size
        self.lattice_size = (2, 2)  # Change this to your desired lattice size
        # Create the graph and sublattice
        self.graph, self.subl = create_graph_from_ham('2dsquare', self.lattice_size, 'Neel')
        # Generate full basis configurations
        n_nodes = len(self.graph.nodes)
        self.full_basis_configs = (np.array([[int(x) for x in format(i, f'0{n_nodes}b')] for i in range(2**(n_nodes))]) * 2 - 1)
        # Create graph tuples
        self.full_basis_gt = create_graph_tuples(self.full_basis_configs, self.graph, self.subl)
        # Initialize the model with given hyperparameters
        hyperparams = {"hidden_size": 128, "output_emb_size": 64}
        self.model = initialize_NQS_model_fromhyperparams('GNN2simple', hyperparams)
        self.sampler = MCMCSampler(self.model, self.full_basis_gt[0])
        self.wave_function_on_full = [self.sampler.evaluate_model(gt) for gt in self.full_basis_gt]
        freq_as_exact_amplitudes = [np.abs(wf)**2 for wf in self.wave_function_on_full]
        freq_as_exact_amplitudes /= np.sum(freq_as_exact_amplitudes)
        self.exact_energy = stochastic_energy(self.sampler, self.graph, self.full_basis_gt, freq_as_exact_amplitudes)
        amplitudes = np.array(self.wave_function_on_full)
        self.psi_full = state_from_config_amplitudes(self.full_basis_configs, amplitudes)
        self.Hamiltonian = qu.ham_heis_2D(
            self.lattice_size[0], self.lattice_size[1], j=1.0, bz=0, cyclic=True, parallel=False
        )
        self.exact_quimb_energy = (self.psi_full.H @ self.Hamiltonian @ self.psi_full) / (self.psi_full.H @ self.psi_full)
        # Initialize variables for energy trajectory
        self.energy_trajectory = []

    def test_energy_trajectory(self):
            # Verify that exact energy and quimb energy are the same
        self.assertTrue(
                np.allclose(self.exact_energy[0], self.exact_quimb_energy),
                f"Exact energy ({self.exact_energy}) and quimb energy ({self.exact_quimb_energy}) do not match.")  
        # Initialize graph tuples for variance computations
        num_samples = 8  # Number of MCMC samples
        graph_tuples_v = initialize_graph_tuples(num_samples, self.graph, self.subl)
        num_iterations = 200  # Number of iterations to perform
        for _ in range(num_iterations):
            n_sites = len(graph_tuples_v[0].nodes[:, 0])
            # Perform MCMC updates
            graph_tuples_v, coeff_var_on_var = zip(*[
                self.sampler.monte_carlo_update(2, gt, 'var') for gt in graph_tuples_v
            ])
            # Create amplitude frequencies from graph tuples
            _, freq_var = create_amplitude_frequencies_from_graph_tuples(graph_tuples_v, coeff_var_on_var)
            # Generate unique graph tuples
            unique_tuples_var = generate_graph_tuples_configs(
                graph_tuples_v[0], sparse_list_to_configs(freq_var.indices[:, 0], n_sites)
            )
            freq_ampl = np.array(freq_var.values) / len(graph_tuples_v)
            # Compute stochastic energy
            stoch_energy = stochastic_energy(self.sampler, self.graph, unique_tuples_var, freq_ampl)
            self.energy_trajectory.append(stoch_energy[0].numpy())
        # Compute standard deviation series
        offset = 0  # Starting point for standard deviation calculation
        std_exact_series, std_mean_series = compute_energy_std_series(
            self.energy_trajectory, self.exact_energy[0], offset
        )
        # Save the plot to a file
        plot_file_path = 'energy_std_plot.png'  # You can customize the file path
        plot_std_series(std_exact_series, std_mean_series, offset, plot_file_path)
        # Assert that the standard deviation decreases over time
        self.assertLess(
            std_exact_series[-20], std_exact_series[2],
            "Standard deviation did not decrease as expected."
        )

def compute_energy_std_series(energy_trajectory, exact_energy, offset=0):
    """
    Compute the standard deviation of the energy trajectory with respect to the exact energy
    and the mean of the trajectory up to each point (starting after offset).
    """
    if len(energy_trajectory) <= offset:
        raise ValueError("Not enough data points to compute statistics after offset.")
    energy_trajectory = np.array(energy_trajectory)
    std_exact_series = []
    std_mean_series = []
    for i in range(offset, len(energy_trajectory)):
        current_trajectory = energy_trajectory[offset:i+1]
        # Standard deviation with respect to exact energy
        std_exact = np.sqrt(np.mean((current_trajectory - exact_energy)**2))
        std_exact_series.append(std_exact)
        # Rolling mean
        mean_estimate = np.mean(current_trajectory)
        # Standard deviation with respect to rolling mean
        std_mean = np.std(current_trajectory - mean_estimate)
        std_mean_series.append(std_mean)
    return std_exact_series, std_mean_series

def plot_std_series(std_exact_series, std_mean_series, offset, file_path):
    import matplotlib.pyplot as plt
    iterations = np.arange(offset, offset + len(std_exact_series))
    plt.figure(figsize=(10, 6))
    plt.loglog(iterations, std_exact_series, label='std_exact', color='b')
    plt.loglog(iterations, std_mean_series, label='std_mean', color='r')
    # Plot 1/sqrt(x) for reference
    plt.loglog(iterations, std_exact_series[0] * (1 / iterations)**0.5, label='1/sqrt(x)', linestyle='--', color='g')
    plt.xlabel('Iterations')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation vs Iterations (Log-Log Scale)')
    plt.legend()
    plt.grid(True)
    # Save the plot to the specified file path
    plt.savefig(file_path)
    plt.close()


if __name__ == '__main__':
    unittest.main()