import argparse
import cProfile
import pstats
from datetime import datetime
# Import necessary modules from your simulation and initialization package
import argparse
from compgraph.training import outer_training
from simulation.initializer import create_graph_from_ham, format_hyperparams_to_string, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
import os
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="Run simulation with hyperparameters")
    parser.add_argument('--graphType', type=str, default='2dsquare')
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--sublattice', type=str, default='Neel')
    parser.add_argument('--beta', type=float, default=0.07)
    parser.add_argument('--full_size_hilbert', type=str, default='yes')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=7e-5)
    parser.add_argument('--outer_loop', type=int, default=5)
    parser.add_argument('--inner_loop', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--output_emb_size', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    
    hyperparams = {
        'graph_params': {
            'graphType': args.graphType,
            'n': args.n,
            'm': args.m,
            'sublattice': args.sublattice
        },
        'sim_params': {
            'beta': args.beta,
            'full_size_hilbert': args.full_size_hilbert,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'outer_loop': args.outer_loop,
            'inner_loop': args.inner_loop,
            'n_batch': args.n_batch,
        },
        'ansatz': 'GNN2simple',
        'ansatz_params': {
            'hidden_size': args.hidden_size,
            'output_emb_size': args.output_emb_size
        }
    }
    
    # Set up profiling
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    # Run the simulation

    # Initialize the graph and its sublattice encoding
    graph, subl = create_graph_from_ham(
        hyperparams['graph_params']['graphType'],
        (hyperparams['graph_params']['n'], hyperparams['graph_params']['m']),
        hyperparams['graph_params']['sublattice']
    )

    # Generate the full basis configurations for the system
    full_basis_configs = np.array([[int(x) for x in format(i, f'0{len(graph.nodes)}b')] for i in range(2**(len(graph.nodes)))]) * 2 - 1
    lowest_eigenstate_as_sparse = initialize_hamiltonian_and_groundstate(hyperparams['graph_params'], full_basis_configs)

    # Initialize the variational and fixed models
    model_w = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'], hyperparams['ansatz_params'])
    model_fix = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'], hyperparams['ansatz_params'])

    # Generate tuples of graphs for variational training and fixed comparisons
    graph_tuples_v = initialize_graph_tuples(
        hyperparams['sim_params']['n_batch'] * hyperparams['sim_params']['batch_size'],
        graph, subl, hyperparams['sim_params']['full_size_hilbert']
    )
    graph_tuples_fix = initialize_graph_tuples(
        hyperparams['sim_params']['n_batch'] * hyperparams['sim_params']['batch_size'],
        graph, subl, hyperparams['sim_params']['full_size_hilbert']
    )

    # Perform the training simulation
    results = outer_training(
        hyperparams['sim_params']['outer_loop'], hyperparams['sim_params']['inner_loop'],
        subl, graph, hyperparams['sim_params']['batch_size'], lowest_eigenstate_as_sparse,
        hyperparams['sim_params']['beta'], hyperparams['sim_params']['learning_rate'],
        model_w, model_fix, graph_tuples_v, graph_tuples_fix
    )

    profiler.disable()  # Stop profiling
    # Create statistics object and print profiling results
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)
    
    stats.dump_stats('simulation.prof')
    stats.get_stats_profile()

    print("Simulation completed. Results are saved.")

if __name__ == "__main__":
    main()
