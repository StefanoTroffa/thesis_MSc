import tensorflow as tf
import sonnet as snt
from graph_nets import modules
from graph_nets import blocks

# # Define the global model
# class MLPModel_glob(snt.Module):
#     def __init__(self, name=None):
#         super(MLPModel_glob, self).__init__(name=name)
#         self.layer1 = snt.Linear(output_size=1, name='LinearGloblayer')

#     def __call__(self, inputs):
#         out = tf.nn.relu(self.layer1(inputs))
#         return out

# # Define the encoder model
# class MLPModel_4layers(snt.Module):
#     def __init__(self, hidden_layer_size, output_emb_size, name=None):
#         super(MLPModel_4layers, self).__init__(name=name)
#         self.layer1 = snt.Linear(output_size=hidden_layer_size, name='layer1')
#         self.layer2 = snt.Linear(output_size=hidden_layer_size, name='layer2')
#         self.layer3 = snt.Linear(output_size=hidden_layer_size, name='layer3')
#         self.layer4 = snt.Linear(output_size=output_emb_size, name='layer4')

#     def __call__(self, inputs):
#         x = tf.nn.relu(self.layer1(inputs))
#         x = tf.nn.relu(self.layer2(x))
#         x = tf.nn.relu(self.layer3(x))
#         out = tf.nn.relu(self.layer4(x))
#         return out

# # Define the Encoder layer
# class Encoder(modules.GraphNetwork):
#     def __init__(self, hidden_layer_size, output_emb_size, name=None):
#         super(Encoder, self).__init__(
#             edge_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
#             node_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
#             global_model_fn=lambda: MLPModel_glob(),
#             name=name
#         )

#     def __call__(self, inputs):
#         return super(Encoder, self).__call__(inputs)

# # Define the Processor layer
# class ProcessorLayer(modules.GraphNetwork):
#     def __init__(self, hidden_layer_size, output_emb_size, name=None):
#         super(ProcessorLayer, self).__init__(
        
#             edge_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
#             node_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
#             global_model_fn=lambda: MLPModel_glob(),   name=name
#         )

#     def call(self, inputs):
#         updated_graph = super(ProcessorLayer, self).__call__(inputs)
#         return inputs.replace(
#             nodes=inputs.nodes + updated_graph.nodes,
#             edges=inputs.edges + updated_graph.edges,
#             globals=inputs.globals + updated_graph.globals)


# # Define the Decoder
# class Decoder(modules.GraphNetwork):
#     def __init__(self, hidden_layer_size, output_emb_size, name=None):
#         super(Decoder, self).__init__(
#             edge_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
#             node_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
#             global_model_fn=lambda: MLPModel_glob(),
#             name=name
#         )

#     def __call__(self, inputs):
#         return super(Decoder, self).__call__(inputs)

# # Define the Pooling layer
# class PoolingLayer_double(snt.Module):
#     def __init__(self):
#         super(PoolingLayer_double, self).__init__()
#         self.linear = snt.Linear(output_size=2, name='linear_pool')
#         self.global_transform = snt.Linear(output_size=2, name='global_transform')

#     def __call__(self, inputs):
#         pooled_nodes = tf.reduce_sum(inputs.nodes, axis=0)
#         pooled_edges = tf.reduce_sum(inputs.edges, axis=0)
#         pooled_features = tf.concat([pooled_nodes, pooled_edges], axis=0)
#         transformed = self.linear(tf.expand_dims(pooled_features, axis=0))
#         transformed_globals = self.global_transform(0.05 * inputs.globals)
#         out = tf.nn.relu(transformed + transformed_globals)
#         return out

# # Define a comprehensive GNN model
# # @tf.function
# class GNN_double_output_advanced(snt.Module):
#     def __init__(self,hidden_layer_size, output_emb_size, num_layers=tf.constant(3)):
#         super(GNN_double_output_advanced, self).__init__()
#         self.encoder = Encoder(hidden_layer_size, output_emb_size)
#         self.processor = ProcessorLayer( hidden_layer_size, output_emb_size)
#         self.decoder = Decoder(hidden_layer_size, output_emb_size)
#         self.pooling_layer = PoolingLayer_double()
#         self.num_layers=num_layers
#     def __call__(self, inputs):
#         encoded = self.encoder(inputs)
#         for _ in range(self.num_layers):
#             proc= self.processor(encoded)
#         decoded = self.decoder(proc)
#         output = self.pooling_layer(decoded)
#         return output
    
# class GNN_double_output(snt.Module):
#     def __init__(self,hidden_layer_size=tf.constant(128), output_emb_size=tf.constant(64)):
#         super(GNN_double_output, self).__init__()
#         self.encoder = Encoder(hidden_layer_size=hidden_layer_size,output_emb_size=output_emb_size)

#         self.pooling_layer = PoolingLayer_double()
#     @tf.function(reduce_retracing=True)
#     def __call__(self, inputs):
#         encoded = self.encoder(inputs)

#         output = self.pooling_layer(encoded)
#         return output

import tensorflow as tf
import sonnet as snt

# Custom initializer example
initializer = tf.keras.initializers.HeNormal()  # He initialization for ReLU
bias_initializer = tf.keras.initializers.Constant(0.1)  # Small positive value for biases

# Define the global model with custom initialization
class MLPModel_glob(snt.Module):
    def __init__(self, name=None):
        super(MLPModel_glob, self).__init__(name=name)
        self.layer1 = snt.Linear(output_size=1, name='LinearGloblayer', w_init=initializer)

    def __call__(self, inputs):
        out = tf.nn.relu(self.layer1(inputs))
        return out

# Define the encoder model with custom initialization
class MLPModel_4layers(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, name=None):
        super(MLPModel_4layers, self).__init__(name=name)
        self.layer1 = snt.Linear(output_size=hidden_layer_size, name='layer1', w_init=initializer)
        self.layer2 = snt.Linear(output_size=hidden_layer_size, name='layer2', w_init=initializer)
        self.layer3 = snt.Linear(output_size=hidden_layer_size, name='layer3', w_init=initializer)
        self.layer4 = snt.Linear(output_size=output_emb_size, name='layer4', w_init=initializer)

    def __call__(self, inputs):
        x = tf.nn.relu(self.layer1(inputs))
        x = tf.nn.relu(self.layer2(x))
        x = tf.nn.relu(self.layer3(x))
        out = tf.nn.relu(self.layer4(x))
        return out

# Define the Encoder layer with custom initialization
class Encoder(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, name=None):
        super(Encoder, self).__init__(name=name)
        self.edge_model = MLPModel_4layers(hidden_layer_size, output_emb_size)
        self.node_model = MLPModel_4layers(hidden_layer_size, output_emb_size)
        self.global_model = MLPModel_glob()

    def __call__(self, inputs):
        return modules.GraphNetwork(
            edge_model_fn=lambda: self.edge_model,
            node_model_fn=lambda: self.node_model,
            global_model_fn=lambda: self.global_model
        )(inputs)

# Define the Processor layer with custom initialization
class ProcessorLayer(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, name=None):
        super(ProcessorLayer, self).__init__(name=name)
        self.edge_model = MLPModel_4layers(hidden_layer_size, output_emb_size)
        self.node_model = MLPModel_4layers(hidden_layer_size, output_emb_size)
        self.global_model = MLPModel_glob()

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

# Define the Decoder with custom initialization
class Decoder(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, name=None):
        super(Decoder, self).__init__(name=name)
        self.edge_model = MLPModel_4layers(hidden_layer_size, output_emb_size)
        self.node_model = MLPModel_4layers(hidden_layer_size, output_emb_size)
        self.global_model = MLPModel_glob()

    def __call__(self, inputs):
        return modules.GraphNetwork(
            edge_model_fn=lambda: self.edge_model,
            node_model_fn=lambda: self.node_model,
            global_model_fn=lambda: self.global_model
        )(inputs)

# Define the Pooling layer with custom initialization
bias_pool= tf.keras.initializers.Constant(0.8)  

class PoolingLayer_double(snt.Module):
    def __init__(self):
        super(PoolingLayer_double, self).__init__()
        self.linear = snt.Linear(output_size=2, name='linear_pool', w_init=initializer)
        self.global_transform = snt.Linear(output_size=2, name='global_transform', w_init=initializer)

    def __call__(self, inputs):
        pooled_nodes = tf.reduce_sum(inputs.nodes, axis=0)
        pooled_edges = tf.reduce_sum(inputs.edges, axis=0)
        # tf.print("Pooled nodes:", pooled_nodes)
        # tf.print("Pooled edges:", pooled_edges)

        pooled_features = tf.concat([pooled_nodes, pooled_edges], axis=0)
        transformed = self.linear(tf.expand_dims(pooled_features, axis=0))
        transformed_globals = self.global_transform(0.05 * inputs.globals)
        # tf.print("features into relu", transformed + transformed_globals)
        # tf.print("output from relu", tf.nn.relu(transformed + transformed_globals))
        out = tf.nn.elu(transformed + transformed_globals)
        return out
class ProcessorStack(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, num_layers):
        super().__init__()
        self.processors = []
        processor = ProcessorLayer(hidden_layer_size, output_emb_size)

        for _ in range(num_layers):
            self.processors.append(processor)

    def __call__(self, graph):
        for processor in self.processors:
            graph = processor(graph)
        return graph

# Define a comprehensive GNN model with custom initialization
class GNN_double_output_advanced(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, num_layers):
        super(GNN_double_output_advanced, self).__init__()
        self.encoder = Encoder(hidden_layer_size, output_emb_size)
        self.processor = ProcessorStack(hidden_layer_size, output_emb_size, tf.constant(num_layers))
        self.decoder = Decoder(hidden_layer_size, output_emb_size)
        self.pooling_layer = PoolingLayer_double()
    @tf.function(reduce_retracing=True)
    def __call__(self, inputs):
        encoded = self.encoder(inputs)
        proc = self.processor(encoded)

        decoded = self.decoder(proc)
        output = self.pooling_layer(decoded)
        return output
       
class GNN_double_output(snt.Module):
    def __init__(self,hidden_layer_size=tf.constant(128), output_emb_size=tf.constant(64)):
        super(GNN_double_output, self).__init__()
        self.encoder = Encoder(hidden_layer_size=hidden_layer_size,output_emb_size=output_emb_size)

        self.pooling_layer = PoolingLayer_double()
    # @tf.function(reduce_retracing=True)
    def __call__(self, inputs):
        encoded = self.encoder(inputs)

        output = self.pooling_layer(encoded)
        return output

