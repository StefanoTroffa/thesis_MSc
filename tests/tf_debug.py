import tensorflow as tf
import numpy as np
from compgraph.monte_carlo import MCMCSampler, propose_graph_tuple
from compgraph.models import GNN_double_output
from compgraph.useful import create_graph_tuples
from simulation.initializer import create_graph_from_ham

# Define the time_evoluted_config_amplitude function with tf.function
class DebugMCMCSampler(MCMCSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tf.function()
    def time_evoluted_config_amplitude_tf(self, graph_tuple):
        return self.time_evoluted_config_amplitude(graph_tuple)

# Example usage:
if __name__ == '__main__':
    # 1. Define graph parameters
    graph_params = {
        'graphType': '2dsquare',
        'n': 2,
        'm': 2,
        'sublattice': 'Neel'
    }
    
    # 2. Create graph and sublattice
    graph, subl = create_graph_from_ham(
        graph_params['graphType'],
        (graph_params['n'], graph_params['m']),
        graph_params['sublattice']
    )
    
    # 3. Define simulation parameters
    sim_params = {
        'beta': 0.05,
        'batch_size': 1,
    }
    
    # 4. Initialize the model
    model = GNN_double_output(32, 16)
    
    # 5. Create graph tuples
    graph_tuples = create_graph_tuples(np.array([[1, -1, 1, -1]]), graph, subl)
    
    # 6. Initialize the DebugMCMCSampler
    sampler = MCMCSampler(model, graph_tuples[0], sim_params['beta'], graph)
    
    # 7. Call the tf.function-decorated method
    result = sampler.time_evoluted_config_amplitude_tf(graph_tuples[0])
    print("Result of time_evoluted_config_amplitude_tf:", result)