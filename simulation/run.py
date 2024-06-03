from datetime import datetime
# Import necessary modules from your simulation and initialization package
import argparse
from simulation.data_handling import run_simulation

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
    
    simulation_results = run_simulation(hyperparams)
    print("Simulation completed. Results are saved.")

if __name__ == "__main__":
    main()
