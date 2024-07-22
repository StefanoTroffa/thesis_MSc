import multiprocessing as mp
#from joblib import Parallel, delayed
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import time
import sonnet as snt
from compgraph.cg_repr import graph_tuple_to_config_hamiltonian_product_update
from compgraph.tensor_wave_functions import evaluate_model, time_evoluted_config_amplitude
# import line_profiler
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
    def __init__(self, model, initial_state, beta=None, graph=None, sublattice_encoding=None):
        self.model = model
        self.state = initial_state
        self.beta = beta
        self.graph = graph
        self.sublattice_encoding = sublattice_encoding
    def update_model(self, model):
        self.model = model
    def evaluate_model(self, graph_tuple):
        amplitude, phase = self.model(graph_tuple)[0]
        return tf.complex(real=amplitude, imag=phase)
    
    # @profile
    # @tf.function
    def monte_carlo_update(self, N_sweeps, approach):
        state = self.state
        if approach == 'var':
            psi_old = self.evaluate_model(state)

            for _ in tf.range(N_sweeps):
                proposed_graph_tuple = propose_graph_tuple(state)
                psi_new = self.evaluate_model(proposed_graph_tuple)
                p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.abs(psi_new / psi_old)**2)
                if tf.random.uniform([], dtype=tf.float64) < p_accept:
                    psi_old=psi_new

                    state = proposed_graph_tuple
        elif approach == 'te':
            psi_old = self.time_evoluted_config_amplitude(state)

            for _ in tf.range(N_sweeps):
                proposed_graph_tuple = propose_graph_tuple(state)
                psi_new = self.time_evoluted_config_amplitude(proposed_graph_tuple)
                p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.abs(psi_new / psi_old)**2)

                if tf.random.uniform([], dtype=tf.float64) < p_accept:
                    psi_old=psi_new
                    state = proposed_graph_tuple

        self.state = state
        return state
    def time_evoluted_config_amplitude(self, graph_tuple):
        graph_tuples_nonzero, amplitudes_gt = graph_tuple_to_config_hamiltonian_product_update(graph_tuple, self.graph, self.sublattice_encoding)
        final_amplitude = []
        for i, gt in enumerate(graph_tuples_nonzero):
            amplitude, phase = self.model(gt)[0]
            amplitude *= amplitudes_gt[i]
            complex_coefficient = tf.complex(real=amplitude, imag=phase)
            final_amplitude.append(complex_coefficient)
        beta = -1. * self.beta
        total_amplitude = tf.multiply(beta, tf.reduce_sum(tf.stack(final_amplitude)))
        complex_coefficient = self.evaluate_model(graph_tuple)
        total_amplitude = tf.add(complex_coefficient, total_amplitude)
        return total_amplitude
