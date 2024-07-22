from datetime import datetime
# Import necessary modules from your simulation and initialization package
import argparse
from simulation.data_handling import run_simulation, parse_args,create_directory_structure


def main():
    args = parse_args()
    
    hyperparams = {
        'simulation_type': args.simulation_type,  # Added simulation_type to hyperparams

        'graph_params': {
            'graphType': args.graphType,
            'n': args.n,
            'm': args.m,
            'sublattice': args.sublattice
        },
        'sim_params': {
            'beta': args.beta,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'outer_loop': args.outer_loop,
            'inner_loop': args.inner_loop,
        },
        'ansatz': 'GNN2simple',
        'ansatz_params': {
            'hidden_size': args.hidden_size,
            'output_emb_size': args.output_emb_size
        }
    }
    
    base_path = create_directory_structure(hyperparams)
    print("Simulation directory created:", base_path)

    simulation_results = run_simulation(hyperparams)
    print("Simulation completed. Results are saved.")

if __name__ == "__main__":
    main()

