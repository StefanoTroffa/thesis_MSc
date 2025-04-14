import tensorflow as tf
import networkx as nx
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf
def initialize_graph_tuples_tf_opt(n_configs, graph, sublattice_encoding, full_size_hilbert=None, sz_sector=None):
    """TensorFlow-compatible version of configuration generation with optional Sz sector constraint."""
    num_nodes = graph.number_of_nodes()
    
    if full_size_hilbert == 'yes':
        # Generate full basis using TensorFlow operations
        n_configs = 2 ** num_nodes
        indices = tf.range(n_configs, dtype=tf.int32)
        basis_configs = tf.bitwise.bitwise_and(tf.bitwise.right_shift(indices[:, tf.newaxis], 
                                              tf.range(num_nodes, dtype=tf.int32)), 1)
        basis_configs = tf.cast(basis_configs, tf.float32) * 2 - 1  # Convert to -1/1
    else:
        if sz_sector is not None:
            # sz_sector should be an integer specifying the number of spin ups
            # Generate configurations with fixed number of spin ups
            num_ups = sz_sector
            num_downs = num_nodes - num_ups
            
            # Create arrays of 1s and -1s
            ups = tf.ones((n_configs, num_ups), dtype=tf.float32)
            downs = -tf.ones((n_configs, num_downs), dtype=tf.float32)
            
            # Concatenate and shuffle each row independently
            combined = tf.concat([ups, downs], axis=1)
            batch_indices = tf.tile(tf.expand_dims(tf.range(n_configs), 1), [1, num_nodes])
            shuffle_indices = tf.stack([batch_indices, tf.argsort(tf.random.uniform([n_configs, num_nodes]))], axis=2)
            basis_configs = tf.gather_nd(combined, shuffle_indices)
        else:
            # Generate random configurations using TensorFlow
            basis_configs = tf.cast(tf.random.uniform((n_configs, num_nodes)) > 0.5, tf.float32) * 2 - 1

    # Create graph tuples with pure TensorFlow operations
    return create_graph_tuples_tf_opt(basis_configs, graph, sublattice_encoding)
def create_graph_tuples_tf_opt(configs: tf.Tensor, graph: nx.Graph, sublattice_encoding: tf.Tensor, 
                       global_par: float = 0.05, edge_par: float = 0.5) -> GraphsTuple:
    """Pure TensorFlow implementation preserving original logic."""
    # Validate input dimensions
    num_nodes = graph.number_of_nodes()
    if sublattice_encoding.shape[0] != num_nodes:
        raise ValueError("Sublattice encoding doesn't match graph nodes")

    # Convert graph structure to TensorFlow tensors
    senders, receivers = [], []
    for u, v in graph.edges():
        senders.extend([u, v])  # Bidirectional edges
        receivers.extend([v, u])
    num_edges = len(senders)

    # Create static tensor components
    edge_features = tf.fill([num_edges, 1], tf.constant(edge_par, dtype=tf.float32))
    global_features = tf.constant([[global_par]], dtype=tf.float32)
    sublattice_tensor = tf.convert_to_tensor(sublattice_encoding, dtype=tf.float32)

    # Process configurations (now tf.Tensor input)
    graph_tuples = []
    for i in tf.range(tf.shape(configs)[0]):
        # Original node feature concatenation logic using TensorFlow
        config_slice = tf.reshape(configs[i], [num_nodes, 1])  # [num_nodes, 1]
        # print(config_slice, sublattice_tensor)
        node_features = tf.concat([config_slice, sublattice_tensor], axis=-1)
        
        graph_tuple = GraphsTuple(
            nodes=node_features,
            edges=edge_features,
            globals=global_features,
            senders=tf.constant(senders, dtype=tf.int32),
            receivers=tf.constant(receivers, dtype=tf.int32),
            n_node=tf.constant([num_nodes], dtype=tf.int32),
            n_edge=tf.constant([num_edges], dtype=tf.int32)
        )
        graph_tuples.append(graph_tuple)

    return utils_tf.concat(graph_tuples, axis=0)

def get_single_graph_from_batch(batched_graphs, index):
    """Extract the i-th graph from a batched GraphsTuple."""
    return utils_tf.get_graph(batched_graphs, index)


from typing import Tuple
def precompute_graph_structure(graph: nx.Graph) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert NetworkX graph to tensorflow tensors."""
    edges = list(graph.edges())
    senders = tf.constant([u for u, _ in edges] + [v for _, v in edges], dtype=tf.int32)
    receivers = tf.constant([v for _, v in edges] + [u for u, _ in edges], dtype=tf.int32)
    edge_pairs = tf.constant(edges, dtype=tf.int32)
    return senders, receivers, edge_pairs  

# @tf.function(jit_compile=True)
def update_graph_tuples_config_jit(graph_tuple: GraphsTuple, 
                                configurations: tf.Tensor) -> GraphsTuple:
    """Update node configurations while preserving structure (JIT-safe)."""
    # Ensure static shape and dtype
    # configurations = tf.ensure_shape(configurations, [5,4])
    configurations = tf.cast(configurations, graph_tuple.nodes.dtype)
    # num_nodes_total = graph_tuple.nodes.shape[0]

    # tf.print("configs shape:", tf.shape(configurations))
    # tf.print("template nodes shape:", tf.shape(graph_tuple.nodes))
    print(graph_tuple.nodes[:, 1:])
    # Construct new nodes with original sublattice encoding
    # tf.debugging.assert_equal(
    # tf.shape(configurations)[0]*tf.shape(configurations)[1],
    # tf.shape(graph_tuple.nodes)[0],
    # message="Mismatch in number of nodes vs configurations")
    batch_size, num_nodes = configurations.shape

    tf.ensure_shape(configurations, [batch_size, num_nodes])
    num_nodes_total = graph_tuple.nodes.shape[0]

    flat_configs = tf.reshape(configurations, [num_nodes_total, 1])  # [B * N, 1]

    new_nodes = tf.concat([
        flat_configs,
        graph_tuple.nodes[:, 1:]
    ], axis=1)
    
    return GraphsTuple(
        nodes=new_nodes,
        edges=graph_tuple.edges,
        globals=graph_tuple.globals,
        senders=graph_tuple.senders,
        receivers=graph_tuple.receivers,
        n_node=graph_tuple.n_node,
        n_edge=graph_tuple.n_edge
    ) 
@tf.function(jit_compile=True)
def update_graph_tuple_config_jit(graph_tuple: GraphsTuple, 
                                config: tf.Tensor) -> GraphsTuple:
    """Update node configurations while preserving structure (JIT-safe)."""
    # Ensure static shape and dtype
    config = tf.ensure_shape(config, [graph_tuple.nodes.shape[0]])
    config = tf.cast(config, graph_tuple.nodes.dtype)
    
    # Construct new nodes with original sublattice encoding
    new_nodes = tf.concat([
        tf.reshape(config, [-1, 1]),
        graph_tuple.nodes[:, 1:]
    ], axis=1)
    
    return GraphsTuple(
        nodes=new_nodes,
        edges=graph_tuple.edges,
        globals=graph_tuple.globals,
        senders=graph_tuple.senders,
        receivers=graph_tuple.receivers,
        n_node=graph_tuple.n_node,
        n_edge=graph_tuple.n_edge
    ) 
@tf.function(jit_compile=False)
def create_hamiltonian_batch_xla(base_graph: GraphsTuple, new_configs: tf.Tensor) -> GraphsTuple:
    """XLA-compatible version for creating batched GraphsTuple."""
    # Get static dimensions from base graph
    num_nodes_per = base_graph.n_node[0]
    num_edges_per = base_graph.n_edge[0]
    batch_size = tf.shape(new_configs)[0]
    
    # 1. Construct batched nodes
    sublattice = tf.expand_dims(base_graph.nodes[:, 1:], 0)
    sublattice_batch = tf.broadcast_to(sublattice, [batch_size, tf.shape(sublattice)[1], tf.shape(sublattice)[2]])
    config_with_channel = tf.expand_dims(new_configs, -1)
    batch_nodes = tf.concat([config_with_channel, sublattice_batch], axis=-1)
    
    # 2. Construct batched edges - using tile instead of repeat
    base_edges = tf.reshape(base_graph.edges, [-1, 1])
    batch_edges = tf.tile(base_edges, [batch_size, 1])
    
    # 3. Construct batched connectivity
    offsets = tf.range(batch_size, dtype=tf.int32) * num_nodes_per
    offsets = tf.reshape(offsets, [-1, 1])
    
    senders_per = tf.tile(tf.expand_dims(base_graph.senders, 0), [batch_size, 1])
    receivers_per = tf.tile(tf.expand_dims(base_graph.receivers, 0), [batch_size, 1])
    
    batch_senders = tf.reshape(senders_per + offsets, [-1])
    batch_receivers = tf.reshape(receivers_per + offsets, [-1])
    
    # 4. Construct final GraphsTuple
    return GraphsTuple(
        nodes=tf.reshape(batch_nodes, [-1, 3]),
        edges=batch_edges,
        globals=tf.tile(base_graph.globals, [batch_size, 1]),
        senders=batch_senders,
        receivers=batch_receivers,
        n_node=tf.fill([batch_size], num_nodes_per),
        n_edge=tf.fill([batch_size], num_edges_per)
    )

def create_hamiltonian_batch_jit_r1(base_graph: GraphsTuple, new_configs: tf.Tensor) -> GraphsTuple:
    """Create valid GraphsTuple batch matching GT_Batch structure."""
    # Get static dimensions from base graph
    num_nodes_per = base_graph.n_node[0]
    num_edges_per = base_graph.n_edge[0]
    batch_size = tf.shape(new_configs)[0]
    # print("batch size in create hamiltonian batch jit r1", batch_size)
    # print("num_edges_per in create hamiltonian batch jit r1", num_edges_per)
    # print("num_nodes_per in create hamiltonian batch jit r1", num_nodes_per)
    # 1. Construct batched nodes -------------------------------------------------
    # new_configs: [batch_size, num_nodes]
    # Sublattice features: [num_nodes, 2] → [batch_size, num_nodes, 2]
    sublattice = tf.tile(base_graph.nodes[:, 1:][tf.newaxis], [batch_size, 1, 1])
    batch_nodes = tf.concat([new_configs[:, :, tf.newaxis], sublattice], axis=-1)

    # 2. Construct batched edges -------------------------------------------------
    # Edge features: [num_edges_per, 1] → [batch_size * num_edges_per, 1] using broadcast_to
    base_edges = tf.reshape(base_graph.edges, [-1, 1])  # Ensure correct shape
    # broadcasted_edges = tf.broadcast_to(base_edges, [batch_size, num_edges_per, 1])
    # batch_edges = tf.reshape(broadcasted_edges, [batch_size * num_edges_per, 1])  # Flatten batch dimension
    # batch_edges = tf.tile(base_edges, [batch_size* num_edges_per, 1])  # [B*E, F]
    # More explicit semantics, works in both modes
    batch_edges = tf.repeat(base_edges, repeats=batch_size, axis=0)
    # batch_edges = tf.tile(base_edges, [batch_size, 1])  # [B*E, F] -> This works for eager execution
    # print("batch_edges shape", batch_edges.shape)
    # 3. Construct batched connectivity ------------------------------------------
    # Offset each graph's sender/receiver indices by their position in the batch
    offsets = tf.range(batch_size, dtype=tf.int32) * num_nodes_per
    offsets = tf.reshape(offsets, [-1, 1])  # [batch_size, 1]

    # Replicate edges for each graph in batch
    senders_per = tf.tile(base_graph.senders[tf.newaxis], [batch_size, 1])
    receivers_per = tf.tile(base_graph.receivers[tf.newaxis], [batch_size, 1])

    # Apply offsets and flatten: [batch_size * num_edges_per]
    batch_senders = tf.reshape(senders_per + offsets, [-1])
    batch_receivers = tf.reshape(receivers_per + offsets, [-1])

    # 4. Construct final GraphsTuple ---------------------------------------------
    return GraphsTuple(
        nodes=tf.reshape(batch_nodes, [-1, 3]),  # Flatten to [total_nodes, 3]
        edges=batch_edges,  # Use broadcasted and reshaped edges
        globals=tf.tile(base_graph.globals, [batch_size, 1]),
        senders=batch_senders,
        receivers=batch_receivers,
        n_node=tf.fill([batch_size], num_nodes_per),
        n_edge=tf.fill([batch_size], num_edges_per)
    )
