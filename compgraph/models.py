import tensorflow as tf
import sonnet as snt
import numpy as np
from graph_nets import modules
from graph_nets import blocks

# Function to get initializer with seed
def get_initializer(seed=None):
    """Create a GlorotUniform initializer with an optional seed."""
    return tf.keras.initializers.GlorotUniform(seed=seed)
# Constant initializer stays the same
# bias_initializer = tf.keras.initializers.Constant(0.01)  # Small positive value for biases

# Define the global model with seeded initialization
class MLPModel_glob(snt.Module):
    def __init__(self, seed=None, name=None):
        super(MLPModel_glob, self).__init__(name=name)
        self.layer1 = snt.Linear(output_size=1, name='LinearGloblayer', 
                                w_init=get_initializer(seed))
    
    def __call__(self, inputs):
        out = tf.nn.relu(self.layer1(inputs))
        return out

# Define the global model with seeded initialization
class MLPModel_globv2(snt.Module):
    def __init__(self, seed=None, name=None):
        super(MLPModel_globv2, self).__init__(name=name)
        # Use different seeds for different layers
        layer1_seed = None if seed is None else seed * 10 + 1
        layer2_seed = None if seed is None else seed * 10 + 2
        
        self.layer1 = snt.Linear(output_size=32, name='LinearGloblayer', 
                                w_init=get_initializer(layer1_seed))
        self.layer2 = snt.Linear(output_size=1, name='layer2', 
                                w_init=get_initializer(layer2_seed))
    
    def __call__(self, inputs):
        x = tf.nn.relu(self.layer1(inputs))
        out = tf.nn.relu(self.layer2(x))
        return out

# Define the encoder model with seeded initialization
class MLPModel_4layers(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, seed=None, name=None):
        super(MLPModel_4layers, self).__init__(name=name)
        # Use different seeds for different layers
        layer1_seed = None if seed is None else seed * 10 + 1
        layer2_seed = None if seed is None else seed * 10 + 2
        layer3_seed = None if seed is None else seed * 10 + 3
        layer4_seed = None if seed is None else seed * 10 + 4
        
        self.layer1 = snt.Linear(output_size=hidden_layer_size, name='layer1', 
                                w_init=get_initializer(layer1_seed))
        self.layer2 = snt.Linear(output_size=hidden_layer_size, name='layer2', 
                                w_init=get_initializer(layer2_seed))
        self.layer3 = snt.Linear(output_size=hidden_layer_size, name='layer3', 
                                w_init=get_initializer(layer3_seed))
        self.layer4 = snt.Linear(output_size=output_emb_size, name='layer4', 
                                w_init=get_initializer(layer4_seed))
    
    def __call__(self, inputs):
        x = tf.nn.relu(self.layer1(inputs))
        x = tf.nn.relu(self.layer2(x))
        x = tf.nn.relu(self.layer3(x))
        out = tf.nn.relu(self.layer4(x))
        return out

# Define the Encoder layer with seeded initialization
class Encoder(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, seed=None, name=None):
        super(Encoder, self).__init__(name=name)
        # Use different seeds for different components
        edge_seed = None if seed is None else seed * 10 + 1
        node_seed = None if seed is None else seed * 10 + 2
        global_seed = None if seed is None else seed * 10 + 3
        
        self.edge_model = MLPModel_4layers(hidden_layer_size, output_emb_size, seed=edge_seed)
        self.node_model = MLPModel_4layers(hidden_layer_size, output_emb_size, seed=node_seed)
        self.global_model = MLPModel_globv2(seed=global_seed)
    
    def __call__(self, inputs):
        return modules.GraphNetwork(
            edge_model_fn=lambda: self.edge_model,
            node_model_fn=lambda: self.node_model,
            global_model_fn=lambda: self.global_model
        )(inputs)

# Define the Processor layer with seeded initialization
class ProcessorLayer(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, seed=None, name=None):
        super(ProcessorLayer, self).__init__(name=name)
        # Use different seeds for different components
        edge_seed = None if seed is None else seed * 10 + 1
        node_seed = None if seed is None else seed * 10 + 2
        global_seed = None if seed is None else seed * 10 + 3
        
        self.edge_model = MLPModel_4layers(hidden_layer_size, output_emb_size, seed=edge_seed)
        self.node_model = MLPModel_4layers(hidden_layer_size, output_emb_size, seed=node_seed)
        self.global_model = MLPModel_globv2(seed=global_seed)
    
    def __call__(self, inputs):
        updated_graph = modules.GraphNetwork(
            edge_model_fn=lambda: self.edge_model,
            node_model_fn=lambda: self.node_model,
            global_model_fn=lambda: self.global_model
        )(inputs)
        return inputs.replace(
            nodes=inputs.nodes + updated_graph.nodes,
            edges=inputs.edges + updated_graph.edges,
            globals=inputs.globals + updated_graph.globals)

## Normalized version of the ProcessorLayer
class ScaledResidual(snt.Module):
    def __init__(self, init_scale=1.0 / np.sqrt(2), name=None):
        super().__init__(name=name)
        self.alpha = tf.Variable(init_scale, trainable=True, dtype=tf.float32)

    def __call__(self, x, update):
        return x + self.alpha * update
class ProcessorLayer_norm(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, seed=None, name=None):
        super(ProcessorLayer_norm, self).__init__(name=name)
        # Use different seeds for different components
        edge_seed = None if seed is None else seed * 10 + 1
        node_seed = None if seed is None else seed * 10 + 2
        global_seed = None if seed is None else seed * 10 + 3
        
        self.edge_model = MLPModel_4layers(hidden_layer_size, output_emb_size, seed=edge_seed)
        self.node_model = MLPModel_4layers(hidden_layer_size, output_emb_size, seed=node_seed)
        self.global_model = MLPModel_globv2(seed=global_seed)
        self.gn = modules.GraphNetwork(
            edge_model_fn=lambda: self.edge_model,
            node_model_fn=lambda: self.node_model,
            global_model_fn=lambda: self.global_model
        )
        self.skip = ScaledResidual()
        self.norm_nodes   = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm_edges   = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.norm_globals = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)

    def __call__(self, graphs):
        updated = self.gn(graphs)
        new_nodes   = self.skip(graphs.nodes,   updated.nodes)
        new_edges   = self.skip(graphs.edges,   updated.edges)
        new_globals = self.skip(graphs.globals, updated.globals)
        return graphs.replace(
            nodes   = self.norm_nodes(new_nodes),
            edges   = self.norm_edges(new_edges),
            globals = self.norm_globals(new_globals)
        )
    
# Stack of ProcessorLayerNorms
class ProcessorStackNorm(snt.Module):
    def __init__(self, hidden_size, output_size, num_layers, seed=None, name=None):
        super().__init__(name=name)
        self.layers = []
        for i in range(num_layers):
            layer_seed = None if seed is None else seed * 100 + i
            layer = ProcessorLayer_norm(hidden_size, output_size, seed=layer_seed)
            self.layers.append(layer)

    def __call__(self, graphs):
        for layer in self.layers:
            graphs = layer(graphs)
        return graphs
    
class ProcessorStack(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, num_layers, seed=None):
        super().__init__()
        self.processors = []
        # Use different seeds for different processor layers
        for i in range(num_layers):
            layer_seed = None if seed is None else seed * 100 + i
            processor = ProcessorLayer(hidden_layer_size, output_emb_size, seed=layer_seed)
            self.processors.append(processor)
    
    def __call__(self, graph):
        for processor in self.processors:
            graph = processor(graph)
        return graph

# Define the Decoder with seeded initialization
class Decoder(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, seed=None, name=None):
        super(Decoder, self).__init__(name=name)
        # Use different seeds for different components
        edge_seed = None if seed is None else seed * 10 + 1
        node_seed = None if seed is None else seed * 10 + 2
        global_seed = None if seed is None else seed * 10 + 3
        
        self.edge_model = MLPModel_4layers(hidden_layer_size, output_emb_size, seed=edge_seed)
        self.node_model = MLPModel_4layers(hidden_layer_size, output_emb_size, seed=node_seed)
        self.global_model = MLPModel_globv2(seed=global_seed)
    
    def __call__(self, inputs):
        return modules.GraphNetwork(
            edge_model_fn=lambda: self.edge_model,
            node_model_fn=lambda: self.node_model,
            global_model_fn=lambda: self.global_model
        )(inputs)
    
# Define the Pooling layer with seeded initialization
class PoolingLayer_double(snt.Module):
    def __init__(self, seed=None):
        super(PoolingLayer_double, self).__init__()
        # Use different seeds for different components
        linear_seed = None if seed is None else seed * 10 + 1
        global_seed = None if seed is None else seed * 10 + 2
        
        self.linear = snt.Linear(output_size=2, name='linear_pool', 
                               w_init=get_initializer(linear_seed))
        self.global_transform = snt.Linear(output_size=2, name='global_transform', 
                                        w_init=get_initializer(global_seed))
    
    def __call__(self, inputs):
        pooled_nodes = tf.reduce_sum(inputs.nodes, axis=0)
        pooled_edges = tf.reduce_sum(inputs.edges, axis=0)
        pooled_features = tf.concat([pooled_nodes, pooled_edges], axis=0)
        transformed = self.linear(tf.expand_dims(pooled_features, axis=0))
        transformed_globals = self.global_transform(0.05 * inputs.globals)
        out = tf.nn.elu(transformed + transformed_globals)
        return out


class PoolingLayer_double_batch(snt.Module):
    def __init__(self, seed=None):
        super(PoolingLayer_double_batch, self).__init__()
        # Use different seeds for different components
        linear_seed = None if seed is None else seed * 10 + 1
        global_seed = None if seed is None else seed * 10 + 2
        
        self.linear = snt.Linear(output_size=2, name='linear_pool', 
                               w_init=get_initializer(linear_seed))
        self.global_transform = snt.Linear(output_size=2, name='global_transform', 
                                        w_init=get_initializer(global_seed))
    
    def __call__(self, inputs):
        # Get batch information from the GraphsTuple
        n_node = inputs.n_node
        n_edge = inputs.n_edge
        
        # Create segment IDs for batch-wise aggregation
        node_segment_ids = tf.repeat(tf.range(tf.shape(n_node)[0]), n_node)
        edge_segment_ids = tf.repeat(tf.range(tf.shape(n_edge)[0]), n_edge)
        num_node_segments = tf.shape(n_node)[0]
        num_edge_segments = tf.shape(n_edge)[0]
        
        # Use unsorted_segment_sum with explicit num_segments
        pooled_nodes = tf.math.unsorted_segment_sum(inputs.nodes, node_segment_ids, num_node_segments)
        pooled_edges = tf.math.unsorted_segment_sum(inputs.edges, edge_segment_ids, num_edge_segments)
        
        # Concatenate along feature dimension
        pooled_features = tf.concat([pooled_nodes, pooled_edges], axis=1)  # [batch_size, node_feat+edge_feat]
        
        # Transform features
        transformed = self.linear(pooled_features)
        transformed_globals = self.global_transform(0.05 * inputs.globals)
        
        # Final activation
        out = tf.nn.elu(transformed + transformed_globals)
        return out
class TwoHeadPoolingBatch(snt.Module):
    """Batch-wise pool a GraphsTuple → [amplitude, phase] per graph."""
    def __init__(self, hidden_size: int, seed=None, name=None):
        super().__init__(name=name)
        # seeds for reproducibility
        proj_seed  = None if seed is None else seed*10 + 1
        amp_seed   = None if seed is None else seed*10 + 2
        phase_seed = None if seed is None else seed*10 + 3

        # optional hidden projection before the two heads
        self.proj = snt.Linear(output_size=hidden_size,
                               w_init=get_initializer(proj_seed),
                               name="pool_proj")

        # head that predicts the *amplitude* ψ₀(s)
        self.amp_head = snt.Linear(output_size=1,
                                   w_init=get_initializer(amp_seed),
                                   name="amp_head")

        # head that predicts the *phase* φ(s)
        self.phase_head = snt.Linear(output_size=1,
                                     w_init=get_initializer(phase_seed),
                                     name="phase_head")

    def __call__(self, graphs):
        # 1) segment‐sum pool nodes & edges per graph
        batch_size = tf.shape(graphs.n_node)[0]

        node_segs = tf.repeat(tf.range(batch_size), graphs.n_node)
        edge_segs = tf.repeat(tf.range(batch_size), graphs.n_edge)

        pooled_nodes = tf.math.unsorted_segment_sum(
            graphs.nodes, node_segs, batch_size
        )
        pooled_edges = tf.math.unsorted_segment_sum(
            graphs.edges, edge_segs, batch_size
        )

        # 2) concat with globals: shape [batch, node_feat+edge_feat+global_feat]
        x = tf.concat([pooled_nodes, pooled_edges, graphs.globals], axis=-1)

        # 3) optional hidden layer
        h = tf.nn.relu(self.proj(x))

        # 4) two independent linear heads (no ELU/ReLU here!)
        amp   = self.amp_head(h)    # ψ₀(s)   — the amplitude
        phase = self.phase_head(h)  # φ(s)    — the angle

        # return shape [batch, 2]
        return tf.concat([amp, phase], axis=-1)       
class GNN_double_output_single(snt.Module):
    def __init__(self, hidden_layer_size=tf.constant(128), output_emb_size=tf.constant(64), seed=None):
        super(GNN_double_output_single, self).__init__()
        # Use different seeds for different components
        encoder_seed = None if seed is None else seed * 10 + 1
        pooling_seed = None if seed is None else seed * 10 + 2
        
        self.encoder = Encoder(hidden_layer_size=hidden_layer_size, 
                             output_emb_size=output_emb_size, 
                             seed=encoder_seed)
        self.pooling_layer = PoolingLayer_double(seed=pooling_seed)
    
    @tf.function(jit_compile=True)
    def __call__(self, inputs):
        encoded = self.encoder(inputs)
        output = self.pooling_layer(encoded)
        return output
    
class GNN_double_output(snt.Module):
    def __init__(self, hidden_layer_size=tf.constant(128), output_emb_size=tf.constant(64), seed=None):
        super().__init__()
        # Use different seeds for different components
        encoder_seed = None if seed is None else seed * 10 + 1
        pooling_seed = None if seed is None else seed * 10 + 2
        
        self.encoder = Encoder(hidden_layer_size=hidden_layer_size, 
                             output_emb_size=output_emb_size, 
                             seed=encoder_seed)
        self.pooling_layer = PoolingLayer_double_batch(seed=pooling_seed)
    
    @tf.function(jit_compile=True)
    def __call__(self, inputs):
        encoded = self.encoder(inputs)
        output = self.pooling_layer(encoded)
        return output

# Define a comprehensive GNN model with seeded initialization
class GNN_double_output_advanced(snt.Module):
    def __init__(self, hidden_layer_size=tf.constant(128), output_emb_size=tf.constant(64), 
                num_layers=tf.constant(1), seed=None):
        super().__init__()
        # Use different seeds for different components
        encoder_seed = None if seed is None else seed * 10 + 1
        processor_seed = None if seed is None else seed * 10 + 2
        decoder_seed = None if seed is None else seed * 10 + 3
        pooling_seed = None if seed is None else seed * 10 + 4
        
        self.encoder = Encoder(hidden_layer_size, output_emb_size, seed=encoder_seed)
        self.processor = ProcessorStack(hidden_layer_size, output_emb_size, num_layers, seed=processor_seed)
        self.decoder = Decoder(hidden_layer_size, output_emb_size, seed=decoder_seed)
        self.pooling_layer = PoolingLayer_double_batch(seed=pooling_seed)
    
    @tf.function(jit_compile=True)
    def __call__(self, inputs):
        encoded = self.encoder(inputs)
        proc = self.processor(encoded)
        decoded = self.decoder(proc)
        output = self.pooling_layer(decoded)
        return output

# Define a comprehensive GNN model with seeded initialization
class GNN_double_output_advanced_proc_norm(snt.Module):
    def __init__(self, hidden_layer_size=tf.constant(128), output_emb_size=tf.constant(64), 
                num_layers=tf.constant(1), seed=None):
        super().__init__()
        # Use different seeds for different components
        encoder_seed = None if seed is None else seed * 10 + 1
        processor_seed = None if seed is None else seed * 10 + 2
        decoder_seed = None if seed is None else seed * 10 + 3
        pooling_seed = None if seed is None else seed * 10 + 4
        
        self.encoder = Encoder(hidden_layer_size, output_emb_size, seed=encoder_seed)
        self.processor = ProcessorStackNorm(hidden_layer_size, output_emb_size, num_layers, seed=processor_seed)
        self.decoder = Decoder(hidden_layer_size, output_emb_size, seed=decoder_seed)
        self.pooling_layer = PoolingLayer_double_batch(seed=pooling_seed)
    
    # @tf.function(jit_compile=True)
    @tf.function(jit_compile=True)
    def __call__(self, inputs):
        encoded = self.encoder(inputs)
        proc = self.processor(encoded)
        decoded = self.decoder(proc)
        output = self.pooling_layer(decoded)
        return output