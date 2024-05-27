from datetime import datetime
# Import necessary modules from your simulation and initialization package
from simulation.data_handling import run_simulation


# Define hyperparameters
hyperparams = {
    'graph_params': {
        'graphType': '2dsquare',
        'n': 2,
        'm': 2,
        'sublattice': 'Neel'
    },
    'sim_params': {
        'beta': 0.07,
        'full_size_hilbert': 'yes',
        'batch_size': 8,
        'learning_rate': 7e-5,
        'outer_loop': 5,
        'inner_loop': 3,
        'n_batch': 2,
    },
    'ansatz': 'GNN2simple',
    'ansatz_params': {
        'hidden_size': 64,
        'output_emb_size': 16
    }
}


if __name__ == "__main__":

    simulation_results = run_simulation(hyperparams)
    print("Simulation completed. Results are saved.")
