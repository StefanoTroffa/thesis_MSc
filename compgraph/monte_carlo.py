import multiprocessing as mp
#from joblib import Parallel, delayed
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import time
import sonnet as snt
from compgraph.cg_repr import config_hamiltonian_product, graph_tuple_to_config_hamiltonian_product_update
from compgraph.tensor_wave_functions import evaluate_model, time_evoluted_config_amplitude
# import line_profiler
from compgraph.useful import generate_graph_tuples_configs, graph_tuple_toconfig, update_graph_tuple_config, generate_graph_tuples_configs_new
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

def propose_graph_tuple(graph_tuple):
    """
    Propose a new graph tuple by flipping the spin of a randomly selected node.
    Args: graph_tuple (tf.Tensor): The input graph tuple.
    Returns: tf.Tensor: The proposed graph tuple with the spin of one node flipped.
    """
    proposed_nodes = graph_tuple.nodes.numpy().copy()  # Convert to numpy array for mutability
    i = np.random.randint(len(proposed_nodes))  # Choose a random node
    proposed_nodes[i, 0] *= -1  # Flip the spin at this node
    return graph_tuple.replace(nodes=tf.constant(proposed_nodes))


#@tf.function

# @profile
def monte_carlo_update(graph_tuple, gnn, N_sweeps, approach, beta=None, graph=None, sublattice_encoding=None):
    """
    Perform a Monte Carlo update on a single graph tuple using the specified approach.
    
    Args:
        graph_tuple (tf.Tensor): The input graph tuple.
        gnn : The graph neural network model.
        N_sweeps (int): The number of Monte Carlo updates to perform.
        approach (str): The approach to use for the Monte Carlo update. It can be either 'var' for variational or 'te' for time-evoluted.
        beta (float, optional): The inverse temperature for the time-evoluted approach.
        graph (tf.Tensor, optional): The graph for the time-evoluted approach.
        sublattice_encoding (tf.Tensor, optional): The sublattice encoding for the time-evoluted approach.
    
    Returns:
        tf.Tensor: The updated graph tuple.
    """
    if approach == 'var':
        for _ in tf.range(N_sweeps): 
            proposed_graph_tuple = propose_graph_tuple(graph_tuple)
            psi_old = evaluate_model(gnn, graph_tuple)
            psi_new = evaluate_model(gnn, proposed_graph_tuple)
            #print(type(tf.abs(psi_new / psi_old)**2))

            p_accept = tf.minimum(tf.constant(1.0,dtype=tf.float64), tf.abs(psi_new / psi_old)**2)
            #print(graph_tuple.nodes[0:,], p_accept)

            if tf.random.uniform([], dtype=tf.float64) < p_accept:
                graph_tuple = proposed_graph_tuple

    elif approach == 'te':
        for sweep in range(N_sweeps):
            proposed_graph_tuple = propose_graph_tuple(graph_tuple)
            psi_old = time_evoluted_config_amplitude(gnn, beta, graph_tuple, graph, sublattice_encoding)
            psi_new = time_evoluted_config_amplitude(gnn, beta, proposed_graph_tuple, graph, sublattice_encoding)
            #print(type(tf.abs(psi_new / psi_old)**2))
            p_accept = tf.minimum(tf.constant(1.0,dtype=tf.float64), tf.abs(psi_new / psi_old)**2)

            if tf.random.uniform([], dtype=tf.float64) < p_accept:
                graph_tuple = proposed_graph_tuple

    return graph_tuple


#@tf.function
def sequential_monte_carlo_update(graph_tuples, gnn, N_sweeps, approach, beta=None, graph=None, sublattice_encoding=None):
    updated_graph_tuples = [monte_carlo_update(graph_tuple, gnn, N_sweeps,approach, beta, graph, sublattice_encoding) for graph_tuple in graph_tuples]
   
    return updated_graph_tuples

from multiprocessing import Pool
def update(args):
    graph_tuple, gnn, N_sweeps, approach, beta, graph, sublattice_encoding = args
    return monte_carlo_update(graph_tuple, gnn, N_sweeps, approach, beta, graph, sublattice_encoding)

def parallel_monte_carlo_update(graph_tuples, gnn, N_sweeps, approach, beta=None, graph=None, sublattice_encoding=None):
    # Create arguments list for each graph_tuple
    
    args = [(graph_tuple, gnn, N_sweeps, approach, beta, graph, sublattice_encoding) for graph_tuple in graph_tuples]

    # mp.set_start_method('spawn', force=True)
    with Pool() as pool:
        updated_graph_tuples = pool.map(update, args)
    
    return updated_graph_tuples



# class Cache:
#     def __init__(self):
#         self.cache = {}

#     def get(self, key):
#         return self.cache.get(key)

#     def set(self, key, value):
#         self.cache[key] = value

#     def clear(self):
#         self.cache.clear()

# def generate_cache_key(model_name, graph_tuple):
#     graph_tuple_hash = tf.reduce_sum(graph_tuple).numpy().tobytes()
#     return f"{model_name}_{graph_tuple_hash}"

# class MCMCSampler:
#     def __init__(self, model, initial_state, beta=None, graph=None, sublattice_encoding=None):
#         self.model = model
#         self.state = initial_state
#         self.beta = beta
#         self.graph = graph
#         self.sublattice_encoding = sublattice_encoding
#         self.cache = Cache()

#     def evaluate_model(self, graph_tuple):
#         cache_key = generate_cache_key('var', graph_tuple)
#         cached_value = self.cache.get(cache_key)
#         if cached_value is not None:
#             return cached_value
#         amplitude, phase = self.model(graph_tuple)[0]
#         result = tf.complex(real=amplitude, imag=phase)
#         self.cache.set(cache_key, result)
#         return result

#     @profile
#     def monte_carlo_update(self, N_sweeps, approach):
#         state = self.state
#         if approach == 'var':
#             for _ in tf.range(N_sweeps):
#                 proposed_graph_tuple = propose_graph_tuple(state)
#                 psi_old = self.evaluate_model(state)
#                 psi_new = self.evaluate_model(proposed_graph_tuple)
#                 p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.abs(psi_new / psi_old)**2)
#                 if tf.random.uniform([], dtype=tf.float64) < p_accept:
#                     state = proposed_graph_tuple
#         elif approach == 'te':
#             for _ in tf.range(N_sweeps):
#                 proposed_graph_tuple = propose_graph_tuple(state)
#                 psi_old = self.time_evoluted_config_amplitude(state)
#                 psi_new = self.time_evoluted_config_amplitude(proposed_graph_tuple)
#                 p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.abs(psi_new / psi_old)**2)
#                 if tf.random.uniform([], dtype=tf.float64) < p_accept:
#                     state = proposed_graph_tuple
#         self.state = state
#         return state

#     def time_evoluted_config_amplitude(self, graph_tuple):
#         cache_key = generate_cache_key('te', graph_tuple)
#         cached_value = self.cache.get(cache_key)
#         if cached_value is not None:
#             return cached_value
        
#         graph_tuples_nonzero, amplitudes_gt = graph_tuple_to_config_hamiltonian_product(graph_tuple, self.graph, self.sublattice_encoding)
#         final_amplitude = []
#         for i, gt in enumerate(graph_tuples_nonzero):
#             amplitude, phase = self.model(gt)[0]
#             amplitude *= amplitudes_gt[i]
#             complex_coefficient = tf.complex(real=amplitude, imag=phase)
#             final_amplitude.append(complex_coefficient)
        
#         beta = -1. * self.beta
#         total_amplitude = tf.multiply(beta, tf.reduce_sum(tf.stack(final_amplitude)))
#         complex_coefficient = self.evaluate_model(graph_tuple)
#         total_amplitude = tf.add(complex_coefficient, total_amplitude)

#         self.cache.set(cache_key, total_amplitude)
#         return total_amplitude
#     def update_model(self, model):
#         self.model = model
#         self.cache.clear()

"""
def parallel_monte_carlo_update(graph_tuples, gnn, N_sweeps, approach, beta=None, graph=None, sublattice_encoding=None):
    def update(graph_tuple):
        return monte_carlo_update(graph_tuple, gnn, N_sweeps, approach, beta, graph, sublattice_encoding)
    
    with tf.device('/GPU:0'):  # Ensure GPU usage
        updated_graph_tuples = Parallel(n_jobs=-1)(delayed(update)(graph_tuple) for graph_tuple in graph_tuples)
    
    return updated_graph_tuples

"""


class MCMCSampler:
    def __init__(self, model, current_tuple, beta=None, graph=None, initialized=False):
        self.model = model
        self.tuple = current_tuple
        self.beta = beta
        self.graph = graph
        # self.sublattice_encoding = sublattice_encoding
        self.initialized=initialized
        if not self.initialized:
            self.model(current_tuple)
            self.initialized=True 
    def update_model(self, model):
        self.model = model
    def evaluate_model(self, graph_tuple):
        amplitude, phase = self.model(graph_tuple)[0]
        return tf.complex(real=amplitude* tf.cos(phase), imag=amplitude * tf.sin(phase))
    def time_evoluted_config_amplitude(self, graph_tuple):
        graph_tuples_nonzero, amplitudes_gt = graph_tuple_to_config_hamiltonian_product_update(graph_tuple, self.graph)
        final_amplitude = []
        for i, gt in enumerate(graph_tuples_nonzero):
            amplitude, phase = self.model(gt)[0]
            amplitude *= amplitudes_gt[i]
            complex_coefficient = tf.complex(real=amplitude* tf.cos(phase), imag=amplitude * tf.sin(phase))
            final_amplitude.append(complex_coefficient)
        beta = -1. * self.beta
        total_amplitude = tf.multiply(beta, tf.reduce_sum(tf.stack(final_amplitude)))
        complex_coefficient = self.evaluate_model(graph_tuple)
        total_amplitude = tf.add(complex_coefficient, total_amplitude)
        return total_amplitude

    # @profile
    # @tf.function
    def monte_carlo_update(self, N_sweeps, graph_tuple, approach):
        self.tuple = graph_tuple
        state=self.tuple
        if approach == 'var':
            psi = self.evaluate_model(state)

            for _ in tf.range(N_sweeps):
                proposed_graph_tuple = propose_graph_tuple(state)
                psi_new = self.evaluate_model(proposed_graph_tuple)
                p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.abs(psi_new / psi)**2)
                if tf.random.uniform([], dtype=tf.float64) < p_accept:
                    psi=psi_new

                    state = proposed_graph_tuple
        elif approach == 'te':
            psi = self.time_evoluted_config_amplitude(state)

            for _ in tf.range(N_sweeps):
                proposed_graph_tuple = propose_graph_tuple(state)
                psi_new = self.time_evoluted_config_amplitude(proposed_graph_tuple)
                p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.abs(psi_new / psi)**2)

                if tf.random.uniform([], dtype=tf.float64) < p_accept:
                    psi=psi_new
                    state = proposed_graph_tuple

        self.tuple = state
        return state, psi


def stochastic_energy(model_var:MCMCSampler, graph, graph_tuple_configs, frequencies=None, J2=None):
    energy=0.+0.j
    local_energies=[]
    # If frequencies are not provided, assume uniform sampling (all frequencies set to 1/N)
    if frequencies is None:
        frequencies = tf.ones(len(graph_tuple_configs), dtype=tf.complex128)/len(graph_tuple_configs)
    if J2==None:
        J2=0.
    for idx, gt in enumerate(graph_tuple_configs):
        temp = 0. + 0.j
        # Evaluate the wavefunction for the current graph tuple
        psi_coeff = model_var.evaluate_model(gt)

        # Convert graph tuple to configuration
        config_gt = graph_tuple_toconfig(gt)

        # Get the non-zero Hamiltonian product configurations and their coefficients
        configurations_nonzero, coefficients = config_hamiltonian_product(config_gt, graph, J2)

        # Generate graph tuples for the non-zero configurations
        graph_tuples_nonzero = generate_graph_tuples_configs_new(gt, configurations_nonzero)

        # Loop over the non-zero configurations to calculate their contributions to the energy
        for idx_nonzero, nonzero_gt in enumerate(graph_tuples_nonzero):
            # Evaluate the model's wavefunction for the non-zero graph tuple
            psi_coeff_nonzero = model_var.evaluate_model(nonzero_gt)

            # Add the contribution to the energy: coefficient * (psi(s') / psi(s))
            temp += coefficients[idx_nonzero] * (psi_coeff_nonzero / psi_coeff)

        # Accumulate energy with frequencies
        energy += temp * frequencies[idx]
        local_energies.append(temp)

    return energy, local_energies


def stochastic_overlap(sampler_var, sampler_te, coeff_te_on_te, unique_tuples_var, unique_tuples_te, frequencies_var, frequencies_te):
    """
    Compute the stochastic overlap between the wavefunctions of sampler_var and sampler_te.
    
    Args:
        sampler_var: MCMC sampler for the wavefunction psi.
        sampler_te: MCMC sampler for the wavefunction phi.
        graph_tuples_var: Graph tuples for sampler_var.
        graph_tuples_te: Graph tuples for sampler_te.
        frequencies_var: Frequencies for psi (sampler_var).
        frequencies_te: Frequencies for phi (sampler_te).
    
    Returns:
        The stochastic overlap between psi and phi.
    """
    overlap_psi_phi = 0.0 + 0.j
    overlap_phi_psi = 0.0 + 0.j
    norm_psi = 0.0 + 0.j
    norm_phi = 0.0 + 0.j
    
    # Iterate through graph tuples and accumulate overlap terms
    for idx, graph_tuple_var in enumerate(unique_tuples_var):
        psi_coeff = sampler_var.evaluate_model(graph_tuple_var)
        phi_coeff= sampler_te.time_evoluted_config_amplitude(graph_tuple_var)
        # Compute the ratio for psi -> phi and vice versa
        overlap_psi_phi += (phi_coeff / psi_coeff) * frequencies_var[idx]
        norm_psi += frequencies_var[idx]

    for idx, graph_tuple_te in enumerate(unique_tuples_te):
        psi_coeff=sampler_var.evaluate_model(graph_tuple_te)
        phi_coeff = coeff_te_on_te[idx]
        overlap_phi_psi += (psi_coeff / phi_coeff) * frequencies_te[idx]
        norm_phi += frequencies_te[idx]
        # Accumulate norms
        

    # Compute the final overlap using the formula
    overlap = (overlap_phi_psi*overlap_psi_phi) / (norm_psi * norm_phi)
    print(f"norms: phi{norm_phi}, psi:{norm_psi}")
    return overlap
def copy_and_perturb_weights(sampler_var, sampler_te, perturbation_scale=1e-4, show_changes=False):
    """
    Copy the parameters from sampler_var.model to sampler_te.model and perturb the weights
    to make sure they are different.
    
    Args:
        sampler_var: The original sampler whose model weights are copied.
        sampler_te: The target sampler to which the weights are copied and perturbed.
        perturbation_scale: The standard deviation of the normal noise added as a perturbation.

    Returns:
        sampler_te: The sampler with the perturbed weights.
    """
    # # Copy the weights
    # for var_param, te_param in zip(sampler_var.model.trainable_variables, sampler_te.model.trainable_variables):
    #     te_param.assign(var_param)

    # Apply a small perturbation to sampler_te's weights to ensure the models are different
    if show_changes:
            
        for param in sampler_var.model.trainable_variables[:1]:
            print('before var param',param)
            
    for param in sampler_te.model.variables:
        perturbation = tf.random.normal(param.shape, mean=0.0, stddev=perturbation_scale, dtype=param.dtype)  # Match dtype
        # print("before pert",param)
        param.assign_add(perturbation)
        # print('after',param)
    if show_changes:

        for param in sampler_var.model.trainable_variables[:1]:
            print('after',param)
                    
    return sampler_te
def stochastic_gradients(sampler_var,sampler_te, unique_tuples_var,freq_ampl_var):
    with tf.GradientTape() as tape:
        tape.watch(sampler_var.model.trainable_variables)
        psi_terms= tf.stack([sampler_var.evaluate_model(gt) for gt in unique_tuples_var])
        phi_terms= tf.stack([sampler_te.time_evoluted_config_amplitude(gt) for gt in unique_tuples_var])
        # print(psi_terms)

        log_psi=tf.math.log(tf.math.conj(psi_terms))
        ratio_phi_psi = tf.stop_gradient(phi_terms / psi_terms)
        # print(ratio_phi_psi, phi_terms)
        # log_psi_gradients=tape.gradient(log_psi, sampler_var.model.trainable_variables)
        stoch_estimation=freq_ampl_var*log_psi -ratio_phi_psi*log_psi*freq_ampl_var/tf.reduce_sum(ratio_phi_psi*freq_ampl_var)
        stoch_gradients=tape.gradient(tf.reduce_sum(stoch_estimation), sampler_var.model.trainable_variables)
        # print(stoch_gradients[0])
    return stoch_gradients    