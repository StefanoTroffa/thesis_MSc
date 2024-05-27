import multiprocessing as mp
from joblib import Parallel, delayed
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
def monte_carlo_update(graph_tuple, gnn, N_sweeps):
    """Perform N_sweeps Monte Carlo updates on a single graph tuple and its corresponding configuration."""
    for _ in range(N_sweeps):
        # Propose a new graph tuple
        proposed_nodes = graph_tuple.nodes.numpy().copy()  # Convert to numpy array for mutability
        i = np.random.randint(len(proposed_nodes))  # Choose a random node
        proposed_nodes[i, 0] *= -1  # Flip the spin at this node

        # Create a new graph tuple with the proposed nodes
        proposed_graph_tuple = graph_tuple.replace(nodes=tf.constant(proposed_nodes))

        # Calculate the acceptance probability
        psi_old = gnn(graph_tuple)[0][0]
        psi_new = gnn(proposed_graph_tuple)[0][0]
        p_accept = min(1, tf.abs(psi_new / psi_old)**2)

        # Accept or reject the new graph tuple
        if np.random.rand() < p_accept:
            graph_tuple = proposed_graph_tuple
            


    return graph_tuple

###FIND OUT WHY THIS DOES NOT WORK
def parallel_monte_carlo_update(graph_tuples, gnn, N_sweeps):
    """Perform N_sweeps Monte Carlo updates on each graph tuple in a batch."""
    # Apply the monte_carlo_update function to each graph tuple in the batch
    updated_graph_tuples = Parallel(n_jobs=-1)(delayed(monte_carlo_update)(graph_tuple, gnn, N_sweeps) for graph_tuple in graph_tuples)

    return updated_graph_tuples

def sequential_monte_carlo_update(graph_tuples, gnn, N_sweeps):
    updated_graph_tuples_and_configs = [monte_carlo_update(graph_tuple, gnn, N_sweeps) for graph_tuple in graph_tuples]
    updated_graph_tuples, updated_configurations = zip(*updated_graph_tuples_and_configs)
    return list(updated_graph_tuples), list(updated_configurations)
def process_batch(batch):
    graph_tuple, gnn, N_sweeps = batch
    return monte_carlo_update(graph_tuple, gnn, N_sweeps)


"""
def process_batch(batch):
    graph_tuple, gnn, N_sweeps = batch
    return monte_carlo_update(graph_tuple, gnn, N_sweeps)

if __name__ == '__main__':
    # Prepare your batches (replace this with your actual batches)
    batches = [(graph_tuple_1, gnn, N_sweeps), (graph_tuple_2, gnn, N_sweeps), ...]

    # Create a pool of workers
    with Pool(processes=4) as pool:  # Adjust the number of processes as needed
        results = pool.map(process_batch, batches)
#updated_graph_tuples = parallel_monte_carlo_update(graph_tuples, gnn, 9)
print(updated_graph_tuples[0].nodes)
"""
