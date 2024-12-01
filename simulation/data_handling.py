import logging
import sys
import pickle
import numpy as np
import os
import subprocess
import platform
from datetime import datetime
import pandas as pd
from compgraph.training import outer_training, outer_training_mc
from simulation.initializer import create_graph_from_ham, format_hyperparams_to_string, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
import os
import argparse
import tensorflow as tf
def parse_args():
    parser = argparse.ArgumentParser(description="Run simulation with hyperparameters")
    
    # General parameters
    parser.add_argument('--simulation_type', type=str, default='VMC', choices=['fsh', 'VMC'], help="Type of simulation")

    # Graph parameters
    graph_parser = parser.add_argument_group('graph_params')
    graph_parser.add_argument('--graphType', type=str, default='2dsquare')
    graph_parser.add_argument('--n', type=int, default=2)
    graph_parser.add_argument('--m', type=int, default=2)
    graph_parser.add_argument('--sublattice', type=str, default='Neel')
    
    # Simulation parameters
    sim_parser = parser.add_argument_group('sim_params')
    sim_parser.add_argument('--beta', type=float, default=0.07)
    sim_parser.add_argument('--full_size_hilbert', type=str, default='VMC',
                             help="Only for default simulation type")
    sim_parser.add_argument('--batch_size', type=int, default=32)
    sim_parser.add_argument('--learning_rate', type=float, default=7e-5)
    sim_parser.add_argument('--outer_loop', type=int, default=500)
    sim_parser.add_argument('--inner_loop', type=int, default=5)    
    # Ansatz parameters
    ansatz_parser = parser.add_argument_group('ansatz_params')
    ansatz_parser.add_argument('--K_layer', type=int, default=None, help="Number of layers in the ansatz (optional)")

    ansatz_parser.add_argument('--hidden_size', type=int, default=128)
    ansatz_parser.add_argument('--output_emb_size', type=int, default=64)
    
    return parser.parse_args()

def get_short_keys():
    return {
        'ansatz': 'ans',
        'ansatz_params': {
            'K_layer': 'K',
            'hidden_size': 'hs',
            'output_emb_size': 'oes',
        },
        'graph_params': {
            'graphType': 'gt',
            'n': '0',
            'm': '0',
            'sublattice': 'sl',
        },
        'sim_params': {
            'beta': 'beta',
            'batch_size': 'bs',
            'learning_rate': 'lr',
            'outer_loop': 'ol',
            'inner_loop': 'il',
        },
    }

def generate_info_string(params, short_keys=None):
    """
    Generate a string representation of the parameters based on short keys.

    Args:
        params (dict): Dictionary of parameters.
        short_keys (dict): Dictionary of short keys mapping.

    Returns:
        str: Concatenated string of parameters.
    """
    if short_keys is None:
        short_keys = get_short_keys()
    
    info_parts = []
    for key, value in params.items():
        if key in short_keys and value is not None:
            if isinstance(value, dict):
                info_parts.append(generate_info_string(value, short_keys[key]))
            else:
                info_parts.append(f"{short_keys[key]}_{value}")
    return '_'.join(info_parts)

def create_directory_structure(hyperparams):
    """
    Create the directory structure based on hyperparameters and simulation type.

    Args:
        hyperparams (dict): Hyperparameters for the simulation.

    Returns:
        str: The base path of the created directory structure.
    """
    # Generate the date string
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    ansatz_info = generate_info_string({'ansatz': hyperparams['ansatz'], 'ansatz_params': hyperparams['ansatz_params']})
    graph_info = f"{hyperparams['graph_params']['graphType']}_0{hyperparams['graph_params']['n']}_0{hyperparams['graph_params']['m']}_{hyperparams['graph_params']['sublattice']}"
    sim_params_info_parts = generate_info_string(hyperparams['sim_params'], get_short_keys()['sim_params']).split('_')
    simulation_type = hyperparams['simulation_type']
    if simulation_type == 'VMC':
        sim_params_info_parts.append('VMC')
    else:
        sim_params_info_parts.append('fsh')

    sim_params_info = '_'.join(sim_params_info_parts)

    # Include the date after 'system_Heisenberg'
    base_path = os.path.join('simulation_results', f'system_Heisenberg_{date_str}', graph_info, sim_params_info, ansatz_info)

    os.makedirs(base_path, exist_ok=True)

    print(f"Results will be saved to {base_path}")
    return base_path

def get_git_hash():
    """ Retrieves the current Git commit hash. "
    
    Returns:
        str: Current Git commit hash or a message indicating no repository.
    """
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
    """ Log detailed hyperparameters used in the simulation. 
    Args:   
    hyperparams (dict): Dictionary of hyperparameters.
    """
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
    """ Log the results of the simulation. 
    
    Args:
        results (dict): Dictionary of results.
        The results also have some "time" dependent vectors such as Energy or Overlap in time.
        we want to print only the last relevant bit of these time vectors 
    """
    logging.info("========= SIMULATION RESULTS ==========")
    for key, value in results.items():
        if isinstance(value, (list, tuple)) and len(value) > 1:
            value = value[-1]
        logging.info(f"{key}: {value}")
    logging.info("=======================================")



def setup_logging(base_path, hyperparams):
    """
    Set up logging for the simulation.

    Args:
        base_path (str): Base directory path for logs.
        hyperparams (dict): Dictionary of (hyper)parameters of the simulation

    Returns:
        logging.Logger: Configured logger.
    """
    log_directory = os.path.join(base_path, 'logs')
    os.makedirs(log_directory, exist_ok=True)
    log_filename = os.path.join(log_directory, "simulation.log")

    # Set up logging handlers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename)
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
    """
    Run the simulation based on the provided hyperparameters and simulation type.

    Args:
        hyperparams (dict): Dictionary of hyperparameters.

    Returns:
        pd.DataFrame: DataFrame containing the results of the simulation.
    """
    # Setup directories and logging
    base_path = create_directory_structure(hyperparams)
    logger = setup_logging(base_path, hyperparams)

    logger.info("Starting simulation...")
    # Initialize the graph and its sublattice encoding
    graph, subl = create_graph_from_ham(
        hyperparams['graph_params']['graphType'],
        (hyperparams['graph_params']['n'], hyperparams['graph_params']['m']),
        hyperparams['graph_params']['sublattice']
    )
    logger.info(f"Graph and sublattice initialized: {graph}, {subl}")
    if hyperparams['graph_params']['n']*hyperparams['graph_params']['m']<17:
        # Generate the full basis configurations for the system
        full_basis_configs = np.array([[int(x) for x in format(i, f'0{len(graph.nodes)}b')] for i in range(2**(len(graph.nodes)))]) * 2 - 1
        lowest_eigenstate_as_sparse = initialize_hamiltonian_and_groundstate(hyperparams['graph_params'], full_basis_configs)

    # Initialize the variational and fixed models
    model_w = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'], hyperparams['ansatz_params'])
    model_fix = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'],hyperparams['ansatz_params'])
    logger.info("Models initialized.")


    simulation_type=hyperparams['simulation_type']
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    if simulation_type == 'VMC':
        # Generate tuples of graphs for variational training and fixed comparisons
        graph_tuples_var = initialize_graph_tuples(hyperparams['sim_params']['batch_size'],
            graph, subl)
        graph_tuples_fix = initialize_graph_tuples(hyperparams['sim_params']['batch_size'],
            graph, subl)
        results=outer_training_mc(
            hyperparams['sim_params']['outer_loop'], hyperparams['sim_params']['inner_loop'], graph,hyperparams['sim_params']['beta'],
             hyperparams['sim_params']['learning_rate'],
            model_w, model_fix, graph_tuples_var, graph_tuples_fix, lowest_eigenstate_as_sparse)
        
    else:
        graph_tuples_var = initialize_graph_tuples(hyperparams['sim_params']['batch_size'],graph, subl,"yes")
        graph_tuples_fix = initialize_graph_tuples(hyperparams['sim_params']['batch_size'],graph, subl,"yes")
        results = outer_training(
            hyperparams['sim_params']['outer_loop'], hyperparams['sim_params']['inner_loop'],
            subl, graph, lowest_eigenstate_as_sparse,
            hyperparams['sim_params']['beta'], hyperparams['sim_params']['learning_rate'],
            model_w, model_fix, graph_tuples_var, graph_tuples_fix)
        

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