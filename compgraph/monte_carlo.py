import multiprocessing as mp
#from joblib import Parallel, delayed
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
from compgraph.tensor_wave_functions import evaluate_model, time_evoluted_config_amplitude

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
        for _ in range(N_sweeps): 
            proposed_graph_tuple = propose_graph_tuple(graph_tuple)
            psi_old = evaluate_model(gnn, graph_tuple)
            psi_new = evaluate_model(gnn, proposed_graph_tuple)
            p_accept = min(1, tf.abs(psi_new / psi_old)**2)
            #print(graph_tuple.nodes[0:,], p_accept)

            if np.random.rand() < p_accept:
                graph_tuple = proposed_graph_tuple

    elif approach == 'te':
        for sweep in range(N_sweeps):
            proposed_graph_tuple = propose_graph_tuple(graph_tuple)
            psi_old = time_evoluted_config_amplitude(gnn, beta, graph_tuple, graph, sublattice_encoding)
            psi_new = time_evoluted_config_amplitude(gnn, beta, proposed_graph_tuple, graph, sublattice_encoding)
            p_accept = min(1, tf.abs(psi_new / psi_old)**2)

            if np.random.rand() < p_accept:
                graph_tuple = proposed_graph_tuple

    return graph_tuple



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

    
    with Pool() as pool:
        updated_graph_tuples = pool.map(update, args)
    
    return updated_graph_tuples

"""
def parallel_monte_carlo_update(graph_tuples, gnn, N_sweeps, approach, beta=None, graph=None, sublattice_encoding=None):
    def update(graph_tuple):
        return monte_carlo_update(graph_tuple, gnn, N_sweeps, approach, beta, graph, sublattice_encoding)
    
    with tf.device('/GPU:0'):  # Ensure GPU usage
        updated_graph_tuples = Parallel(n_jobs=-1)(delayed(update)(graph_tuple) for graph_tuple in graph_tuples)
    
    return updated_graph_tuples

"""