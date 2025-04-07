import sonnet as snt
from compgraph.models import GNN_double_output_single, GNN_double_output_advanced, GNN_double_output
from compgraph.useful import create_2d_square_graph, create_graph_tuples
import time
import numpy as np
import tensorflow as tf
import quimb as qu
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, time_evoluted_wave_function_on_batch
from compgraph.tensor_wave_functions import montecarlo_logloss_overlap_time_evoluted, sparse_tensor_exp_energy, calculate_sparse_overlap, quimb_vec_to_sparse
import sonnet as snt
from compgraph.useful import copy_to_non_trainable  # Importing custom functions and model class

def format_hyperparams_to_string(hyperparams):
    """
    Concatenate the contents of the hyperparameters dictionary into a single string.

    Args:
    - hyperparams (dict): Nested dictionary containing all hyperparameters.

    Returns:
    - str: A formatted string representation of all hyperparameters.
    """
    parts = []

    # Function to handle nested dictionaries
    def parse_dict(d, prefix=''):
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursive call for nested dictionaries
                parse_dict(value, prefix=prefix + key + '_')
            else:
                # Append the formatted string
                parts.append(f"{value}")

    parse_dict(hyperparams)
    return '_'.join(parts)


def initialize_NQS_model_fromhyperparams(ansatz:str, ansatz_params:dict):
    """
    Initialize the model based on the provided ansatz and its parameters.

    Args:
    - ansatz (str): A string indicating the type of model (ansatz) to initialize.
    - ansatz_params (dict): A dictionary of parameters for the ansatz.

    Returns:
    - An instance of the specified model.
    """
    # Mapping of ansatz strings to model constructors
    model_mapping = {
        'GNN2simple': lambda params: GNN_double_output(tf.constant(params['hidden_size']), tf.constant(params['output_emb_size'])),
        'GNN2adv': lambda params: GNN_double_output_advanced(tf.constant(params['hidden_size']), tf.constant(params['output_emb_size']), tf.constant(params['K_layer'])),
}

    if ansatz in model_mapping:
        return model_mapping[ansatz](ansatz_params)
    else:
        available_models = ', '.join(model_mapping.keys())
        raise ValueError(f"This model cannot be initialized. The available models are: {available_models}")


def neel_state(graph):
    num_sites=len(graph.nodes)
    sublattice_encoding = np.zeros((num_sites, 2))  # Two sublattices
    sublattice_encoding[::2, 0] = 1  # Sublattice 1
    sublattice_encoding[1::2, 1] = 1  # Sublattice 2 
    return sublattice_encoding

def disordered_state(graph):
    """
    Creates a disordered state encoding that maintains the same dimensionality
    as the Néel state (2 features) but without spatial structure.
    All nodes have the same encoding, representing a paramagnetic state.
    """
    num_sites = len(graph.nodes)
    # Use uniform values but keep 2D feature representation for compatibility
    sublattice_encoding = np.zeros((num_sites, 2))
    # Set both channels to 0.5 to represent equal probability of both sublattices
    sublattice_encoding[:, 0] = 0.5
    sublattice_encoding[:, 1] = 0.5
    return sublattice_encoding

def apply_sublattice_encoding(graph, sublattice_type):
    """
    Apply the specified sublattice encoding to the given graph.

    Args:
    - graph (networkx.Graph): The graph to which the sublattice encoding will be applied.
    - sublattice_type (str): The type of sublattice encoding to apply.

    Returns:
    - The graph with sublattice encoding applied.
    """
    # Mapping of sublattice type strings to their corresponding functions
    sublattice_mapping = {
        'Neel': neel_state,
        'Disordered': disordered_state
    }

    if sublattice_type in sublattice_mapping:
        sub_lattice_encoding=sublattice_mapping[sublattice_type](graph)
                # Add 'features' to nodes
        for node in graph.nodes():
            graph.nodes[node]['features'] = sub_lattice_encoding[node]


        return graph, sub_lattice_encoding

    else:
        available_sublattices = ', '.join(sublattice_mapping.keys())
        raise ValueError(f"The specified sublattice type is not implemented. Available types are: {available_sublattices}")


def create_graph_from_ham(geometric_structure:str, lattice_size:tuple, sublattice:str):
    """
    Create a graph based on specified geometric structure.

    Args:
    - geometric_structure (str): The type of geometric structure to create.
    - lattice_size (tuple): The dimensions of the lattice.
    - sublattice (str): sublattice encoding we are testing out

    Returns:
    - A graph object based on the specified geometric structure.
    """
    # Mapping of geometric structure strings to graph creation functions
    graph_mapping = {
        '2dsquare': create_2d_square_graph
    }

    if geometric_structure in graph_mapping:
        graph_initial=graph_mapping[geometric_structure](lattice_size)
        # Add the same 'features' to both directions of each edge
        for edge in graph_initial.edges():
            graph_initial.edges[edge]['features'] = [1.0]  # Example feature for edges

        graph_encoded, sublattice_encoding =apply_sublattice_encoding(graph_initial, sublattice)
        return graph_encoded, sublattice_encoding
    else:
        available_structures = ', '.join(graph_mapping.keys())
        raise NotImplementedError(f"The specified geometric structure is not implemented. Available structures are: {available_structures}")

def initialize_hamiltonian_and_groundstate(graph_params, full_basis_configs):
    """
    Initialize the Hamiltonian using quimb, find its ground state, and convert it to sparse format.
    The function is set to handle different types of graphs based on the provided graph parameters.

    Args:
    - graph_params (dict): Parameters for the graph which includes the type and dimensions.
    - full_basis_configs (array): Full basis configurations for the system.
    - num_nodes (int): Number of nodes in the graph.

    Returns:
    - tuple: Contains the Hamiltonian, its lowest eigenstate as a quimb tensor, and the lowest eigenstate in sparse format.
    
    Raises:
    - ValueError: If the graph type is not supported.
    """
    # Mapping of graph types to their respective Hamiltonian creation functions
    hamiltonian_creators = {
        '2dsquare': lambda n, m: qu.ham_heis_2D(n, m, j=1.0, bz=0, cyclic=True, parallel=False, ownership=None)
    }

    graph_type = graph_params.graphType
    if graph_type in hamiltonian_creators:
        # Create the Hamiltonian for the specified graph type
        Hamiltonian_quimb = hamiltonian_creators[graph_type](graph_params.n, graph_params.m)
    else:
        raise ValueError(f"Unsupported graph type '{graph_type}'. Available types: {', '.join(hamiltonian_creators.keys())}")

    # Calculate the ground state of the Hamiltonian
    lowest_eigenstate = qu.groundstate(Hamiltonian_quimb)

    # Convert the lowest eigenstate to a sparse vector
    lowest_eigenstate_as_sparse = quimb_vec_to_sparse(lowest_eigenstate, full_basis_configs, graph_params.n*graph_params.m)

    return lowest_eigenstate_as_sparse
    
def initialize_graph_tuples(n_configs, graph, sublattice_encoding, full_size_hilbert=None):
    """
    Initialize graph data by generating either full basis configurations for the entire Hilbert space 
    or a specified number of random basis configurations, and creating corresponding graph tuples.

    Args:
    - n_configs (int): Number of configurations to generate (ignored if full_size_hilbert is 'yes').
    - num_sites (int): Number of sites in the graph (total nodes).
    - graph (networkx.Graph): The graph structure on which configurations are based.
    - sublattice_encoding (numpy.array): Array encoding sublattice properties.
    - full_size_hilbert (str): If 'yes', generate the full basis configurations for the system, 
      otherwise generate a specified number of random configurations.

    Returns:
    - list: A list of graph tuples created from the basis configurations.
    """
    if full_size_hilbert == 'yes':
        # Generate full basis configurations for the Hilbert space
        n_configs = 2 ** len(graph.nodes)  # Calculate the total possible states
        basis_configs = np.array([[int(x) for x in format(i, f'0{len(graph.nodes)}b')] for i in range(n_configs)]) * 2 - 1
    else:
        # Generate random basis configurations
        basis_configs = np.random.randint(2, size=(n_configs, len(graph.nodes))) * 2 - 1  # Random spins (-1 or 1)


    # Create graph tuples from the basis configurations
    graph_tuples = create_graph_tuples(basis_configs, graph, sublattice_encoding)

    return graph_tuples
 