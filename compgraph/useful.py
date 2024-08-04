from scipy.sparse import csr_matrix
import numpy as np
from graph_nets import utils_np, utils_tf
import quimb as qu
import networkx as nx
import tensorflow as tf
import sonnet as snt
# import line_profiler
# import atexit
# profile3 = line_profiler.LineProfiler()
# atexit.register(profile3.print_stats)
def node_to_index(graph):
    nodes_idx = {node: i for i, node in enumerate(graph.nodes)}
    return nodes_idx

def visualize_weights(module, prefix=''):
    """
    Recursively list the layers and sub-layers of a TensorFlow/Sonnet module,
    and visualize their weights if they have any.
    """
    if isinstance(module, (tf.Module, tf.keras.layers.Layer, snt.Module)):
        for weight in module.variables:
            print(f"{prefix}{weight.name}: values {weight.shape}")
    return
       

def list_layers(module, prefix=''):
    """
    Recursively list the layers and sub-layers of a TensorFlow/Sonnet module.
    
    Args:
    - module: The module from which to list layers.
    - prefix: A prefix for nested layer names to indicate their hierarchy.
    """
    # Check if the module is an instance of the layers we're interested in.
    # Depending on your TensorFlow/Sonnet version, you might need to adjust these checks.
    if isinstance(module, (tf.Module, tf.keras.layers.Layer, snt.Module)):
        print(f"{prefix}{module.name}: {type(module).__name__}")
        
        # Iterate through all attributes of the module.
        for name, attr in module.__dict__.items():
            # Recursively list layers.
            list_layers(attr, prefix=prefix + '  ')
    return

def create_graph_tuples(configs, graph,sublattice_encoding, global_par=0.05, edge_par=0.5):
    node_features = np.concatenate([configs[:, :, np.newaxis], np.repeat(sublattice_encoding[np.newaxis, :, :], len(configs), axis=0)], axis=2)
    # Get the edge indices
    edge_index = np.array(graph.edges()).T
    edge_index_duplicated = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    
    # Create a list of graph dicts
    graph_tuples = []
    for i in range(len(configs)):
        graph_dict = {
            'globals': np.array([global_par]),
            'nodes': node_features[i],
            'edges': np.full((edge_index_duplicated.shape[1], 1), edge_par),
            'senders': edge_index_duplicated[0],
            'receivers': edge_index_duplicated[1]
        }
        
        # Convert to a GraphsTuple and append to the list
        graph_tuples.append(utils_tf.data_dicts_to_graphs_tuple([graph_dict]))


    return graph_tuples

def update_graph_tuple_config(graph_tuple, config, sublattice_encoding):
    new_node_features = np.concatenate([config[:, np.newaxis], sublattice_encoding], axis=1)
    
    graph_tuple = graph_tuple.replace(nodes=tf.convert_to_tensor(new_node_features, dtype=tf.float64))
    return graph_tuple
def generate_graph_tuples_configs(graph_tuple, configs, sublattice):
    graph_tuples=[]
    for config in configs:
        graph_tuples.append(update_graph_tuple_config(graph_tuple, config, sublattice))
    return graph_tuples

def compare_graph_tuples(graph_tuples1, graph_tuples2):
    if len(graph_tuples1) != len(graph_tuples2):
        return False
    for gt1, gt2 in zip(graph_tuples1, graph_tuples2):
        nodes_equal = tf.reduce_all(tf.equal(gt1.nodes, gt2.nodes))
        edges_equal = tf.reduce_all(tf.equal(gt1.edges, gt2.edges))
        globals_equal = tf.reduce_all(tf.equal(gt1.globals, gt2.globals))
        if not nodes_equal.numpy() or not edges_equal.numpy() or not globals_equal.numpy():

            return False
    return True

def graph_tuple_toconfig(graph_tuple):
    config= graph_tuple.nodes[:, 0].numpy()
    return config

def graph_tuple_list_to_configs_list(graph_tuples):
    configs=[]
    for graph_tuple in graph_tuples:
        configs.append(graph_tuple_toconfig(graph_tuple))
    return configs    

def config_list_to_state_list(configs):
    states=[]
    for config in configs:
        states.append(config_to_state(config))
    return states    

def config_to_state(config):
    psi_list=[]
    for i in config:
        if i==1:
            psi_temp=qu.down()
        else:
            psi_temp=qu.up()    
        psi_list.append(psi_temp)
    return qu.kron(*psi_list)

def state_from_config_amplitudes(configurations, amplitudes):
        # Convert configurations to states and compute the final state as superposition
        states = config_list_to_state_list(configurations)
        scaled_states = [amp * state for amp, state in zip(amplitudes, states)]
        superposition_state = sum(scaled_states)  # Superposition of all states
        return superposition_state

def neel_state(graph):
    num_sites=len(graph.nodes)
    sublattice_encoding = np.zeros((num_sites, 2))  # Two sublattices
    sublattice_encoding[::2, 0] = 1  # Sublattice 1
    sublattice_encoding[1::2, 1] = 1  # Sublattice 2 
    return sublattice_encoding

def create_2d_square_graph(lattice_size):
    G = nx.grid_2d_graph(*lattice_size, periodic=True)
    G = nx.relabel_nodes(G, node_to_index(G))
    return G



def sparse_to_config(row_index, num_sites):
    """
    Convert a row index from a sparse representation back to the spin up and down configuration.
    
    :param row_index: Integer, the row index in the sparse matrix.
    :param num_sites: Integer, the number of sites (spins) in the configuration.
    :return: A numpy array representing the configuration.
    """
    # Calculate the total configurations possible
    total_configs = 2 ** num_sites
    # Find the index in the binary representation
    binary_index = total_configs - row_index - 1
    # Convert index to binary string, filling leading zeros to match the number of sites
    binary_str = format(binary_index, '0{}b'.format(num_sites))
    # Convert binary string to configuration array
    config = np.array([1 if x == '0' else -1 for x in binary_str])
    return config

def sparse_list_to_configs(sparse_indices, num_sites):
    configs=[]
    for row_index in sparse_indices:
        config=sparse_to_config(row_index, num_sites)
        configs.append(config)
    conf_array=np.array(configs)
    return conf_array


def sites_to_sparse(base_config):
    values=[]
    configurations_in_sparse_notation=[]
    for configuration in base_config:
        value=0
        for j in range(len(configuration)):
            b= int((-1*(configuration[j]-1))*2**(len(configuration)-j-2))
            value+=b

        values.append(value)
        #print(value)
        one_hot_vector = csr_matrix(([1], ([0], [value])), shape=(1, 2 ** len(configuration)), dtype=np.int8)
        configurations_in_sparse_notation.append(one_hot_vector)
    return configurations_in_sparse_notation, values

def compare_sonnet_modules(snt1, snt2):
    # Ensure both modules have the same number of variables
    if len(snt1.variables) == len(snt2.variables):
        print("The two modules have the same number of layers parameters")
        
        # Iterate over pairs of variables
        for i, (var1, var2) in enumerate(zip(snt1.variables, snt2.variables)):
            # Compare the values of the variables
            if not tf.reduce_all(tf.equal(var1, var2)):
                print(f"Comparison failed at index {i}, {var1.name} is different than {var2.name}")
                return False
            

    else:
        print("The two modules do not have the same number of variables")
        return False
    return True

def copy_to_non_trainable(module_a, module_b):
    copy_weights = tf.group(*[vb.assign(va) for va, vb in zip(module_a.variables, module_b.variables)])
    
    for var in module_b.variables:
        var._trainable = False 
    return

