import tensorflow as tf
import sonnet as snt
from graph_nets import modules
from graph_nets import blocks

# Define the global model
class MLPModel_glob(snt.Module):
    def __init__(self, name=None):
        super(MLPModel_glob, self).__init__(name=name)
        self.layer1 = snt.Linear(output_size=1, name='LinearGloblayer')

    def __call__(self, inputs):
        out = tf.nn.relu(self.layer1(inputs))
        return out

# Define the encoder model
class MLPModel_4layers(snt.Module):
    def __init__(self, hidden_layer_size, output_emb_size, name=None):
        super(MLPModel_4layers, self).__init__(name=name)
        self.layer1 = snt.Linear(output_size=hidden_layer_size, name='layer1')
        self.layer2 = snt.Linear(output_size=hidden_layer_size, name='layer2')
        self.layer3 = snt.Linear(output_size=hidden_layer_size, name='layer3')
        self.layer4 = snt.Linear(output_size=output_emb_size, name='layer4')

    def __call__(self, inputs):
        x = tf.nn.relu(self.layer1(inputs))
        x = tf.nn.relu(self.layer2(x))
        x = tf.nn.relu(self.layer3(x))
        out = tf.nn.relu(self.layer4(x))
        return out

# Define the Encoder layer
class Encoder(modules.GraphNetwork):
    def __init__(self, hidden_layer_size, output_emb_size, name=None):
        super(Encoder, self).__init__(
            edge_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
            node_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
            global_model_fn=lambda: MLPModel_glob(),
            name=name
        )

    def __call__(self, inputs):
        return super(Encoder, self).__call__(inputs)

# Define the Processor layer
class ProcessorLayer(modules.GraphNetwork):
    def __init__(self, hidden_layer_size, output_emb_size, name=None):
        super(ProcessorLayer, self).__init__(
        
            edge_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
            node_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
            global_model_fn=lambda: MLPModel_glob(),   name=name
        )

    def call(self, inputs):
        updated_graph = super(ProcessorLayer, self).__call__(inputs)
        return inputs.replace(
            nodes=inputs.nodes + updated_graph.nodes,
            edges=inputs.edges + updated_graph.edges,
            globals=inputs.globals + updated_graph.globals)


# Define the Decoder
class Decoder(modules.GraphNetwork):
    def __init__(self, hidden_layer_size, output_emb_size, name=None):
        super(Decoder, self).__init__(
            edge_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
            node_model_fn=lambda: MLPModel_4layers(hidden_layer_size, output_emb_size),
            global_model_fn=lambda: MLPModel_glob(),
            name=name
        )

    def __call__(self, inputs):
        return super(Decoder, self).__call__(inputs)

# Define the Pooling layer
class PoolingLayer_double(snt.Module):
    def __init__(self):
        super(PoolingLayer_double, self).__init__()
        self.linear = snt.Linear(output_size=2, name='linear_pool')
        self.global_transform = snt.Linear(output_size=2, name='global_transform')

    def __call__(self, inputs):
        pooled_nodes = tf.reduce_sum(inputs.nodes, axis=0)
        pooled_edges = tf.reduce_sum(inputs.edges, axis=0)
        pooled_features = tf.concat([pooled_nodes, pooled_edges], axis=0)
        transformed = self.linear(tf.expand_dims(pooled_features, axis=0))
        transformed_globals = self.global_transform(0.05 * inputs.globals)
        out = tf.nn.elu(transformed + transformed_globals)
        return out

# Define a comprehensive GNN model
class GNN_double_output_advanced(snt.Module):
    def __init__(self,hidden_layer_size, output_emb_size, num_layers=3):
        super(GNN_double_output_advanced, self).__init__()
        self.encoder = Encoder(hidden_layer_size, output_emb_size)
        self.processor = ProcessorLayer( hidden_layer_size, output_emb_size)
        self.decoder = Decoder(hidden_layer_size, output_emb_size)
        self.pooling_layer = PoolingLayer_double()
        self.num_layers=num_layers
    def __call__(self, inputs):
        encoded = self.encoder(inputs)
        for _ in range(self.num_layers):
            proc= self.processor(encoded)
        decoded = self.decoder(proc)
        output = self.pooling_layer(decoded)
        return output

    
class GNN_double_output(snt.Module):
    def __init__(self,hidden_layer_size, output_emb_size):
        super(GNN_double_output, self).__init__()
        self.encoder = Encoder(hidden_layer_size=hidden_layer_size,output_emb_size=output_emb_size)

        self.pooling_layer = PoolingLayer_double()

    def __call__(self, inputs):
        encoded = self.encoder(inputs)

        output = self.pooling_layer(encoded)
        return output