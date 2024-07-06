import logging
import sys
import pickle
import numpy as np
import os
import subprocess
import platform
from datetime import datetime
import pandas as pd
from compgraph.training import outer_training
from simulation.initializer import create_graph_from_ham, format_hyperparams_to_string, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
import os

def create_directory_structure(hyperparams):
    # Construct the folder names based on hyperparameters
    ansatz_info = f"{hyperparams['ansatz']}_{hyperparams['ansatz_params']['hidden_size']}_{hyperparams['ansatz_params']['output_emb_size']}"
    system_info = "system_Heisenberg"
    graph_info = f"{hyperparams['graph_params']['graphType']}_0{hyperparams['graph_params']['n']}_0{hyperparams['graph_params']['m']}_{hyperparams['graph_params']['sublattice']}"
    sim_params_info = f"beta_{hyperparams['sim_params']['beta']}_fullHilbert_{hyperparams['sim_params']['full_size_hilbert']}_batch_{hyperparams['sim_params']['batch_size']}_lr_{hyperparams['sim_params']['learning_rate']}_loops_{hyperparams['sim_params']['outer_loop']}_{hyperparams['sim_params']['inner_loop']}"

    # Create the directory path
    base_path = os.path.join('simulation_results',ansatz_info, system_info, graph_info, sim_params_info)
    
    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)

    

    print(f"Results will be saved to {base_path}")
    return base_path

def get_git_hash():
    """ Retrieves the current Git commit hash. """
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "No git repository"

def get_system_info():
    """ Retrieves system information. """
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'python_version': platform.python_version()
    }

def format_system_info(info):
    """ Format system information for logging. """
    return ', '.join(f"{key}: {value}" for key, value in info.items())

def log_hyperparameters(hyperparams):
    """ Log detailed hyperparameters used in the simulation. """
    logging.info("========= HYPERPARAMETERS ==========")
    for key, value in hyperparams.items():
        if isinstance(value, dict):
            logging.info(f"{key}:")
            for subkey, subvalue in value.items():
                logging.info(f"  {subkey}: {subvalue}")
        else:
            logging.info(f"{key}: {value}")
    logging.info("===================================")

def log_results(results):
    """ Log the results of the simulation. """
    logging.info("========= SIMULATION RESULTS ==========")
    for key, value in results.items():
        if isinstance(value, (list, tuple)) and len(value) > 1:
            value = value[-1]
        logging.info(f"{key}: {value}")
    logging.info("=======================================")



def setup_logging(hyperparams, log_directory='logs'):
    str_name = format_hyperparams_to_string(hyperparams)
    log_filename = f"{str_name}.log"

    # Ensure the directory exists
    os.makedirs(log_directory, exist_ok=True)
    full_log_path = os.path.join(log_directory, log_filename)

    # Set up logging handlers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(full_log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Log initial configuration and system info
    logger.info(f"{log_hyperparameters(hyperparams)}")
    logger.info(f"Git hash: {get_git_hash()}")
    logger.info(f"System info: {format_system_info(get_system_info())}")

    return logger




def run_simulation(hyperparams):
    # Setup directories and logging
    base_path = create_directory_structure(hyperparams)
    logger = setup_logging(hyperparams, log_directory=base_path)

    logger.info("Starting simulation...")
    # Initialize the graph and its sublattice encoding
    graph, subl = create_graph_from_ham(
        hyperparams['graph_params']['graphType'],
        (hyperparams['graph_params']['n'], hyperparams['graph_params']['m']),
        hyperparams['graph_params']['sublattice']
    )
    logger.info(f"Graph and sublattice initialized: {graph}, {subl}")

    # Generate the full basis configurations for the system
    full_basis_configs = np.array([[int(x) for x in format(i, f'0{len(graph.nodes)}b')] for i in range(2**(len(graph.nodes)))]) * 2 - 1
    lowest_eigenstate_as_sparse = initialize_hamiltonian_and_groundstate(hyperparams['graph_params'], full_basis_configs)

    # Initialize the variational and fixed models
    model_w = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'], hyperparams['ansatz_params'])
    model_fix = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'], hyperparams['ansatz_params'])
    logger.info("Models initialized.")

    # Generate tuples of graphs for variational training and fixed comparisons
    graph_tuples_var = initialize_graph_tuples(
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
        model_w, model_fix, graph_tuples_var, graph_tuples_fix
    )

    # Collect all results for DataFrame construction
    results_data = {
        "sim_time": results[0],
        "energies": results[1],  # Convert tensors to numpy if necessary
        "loss": results[2],
        "overlap": results[3]
    }

    # Flatten the hyperparameters and merge with results
    flat_hyperparams = flatten_dict(hyperparams)
    simulation_entry = {**flat_hyperparams, **results_data}

    # Create a DataFrame for the single simulation run
    results_df = pd.DataFrame([simulation_entry])

    # Save results to a pickle file in the specified directory
    results_filename = os.path.join(base_path, f"results_{format_hyperparams_to_string(hyperparams)}.pkl")
    results_df.to_pickle(results_filename)
    logger.info(f"{log_results(results_data)}")
    logger.info(f"Results saved to {results_filename}")
    
    return results_df

def flatten_dict(d):
    """
    Flattens a nested dictionary and keeps only the deepest level keys.
    
    Args:
        d (dict): The dictionary to flatten.
        
    Returns:
        dict: A new dictionary with only the deepest keys.
    """
    items = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively call flatten_dict until dictionary values no longer contain nested dictionaries
            deeper_items = flatten_dict(value)
            for subkey, subvalue in deeper_items.items():
                # Only add the deepest layer keys
                if isinstance(subvalue, dict):
                    items.update(flatten_dict(subvalue))
                else:
                    items[subkey] = subvalue
        else:
            items[key] = value
    return items