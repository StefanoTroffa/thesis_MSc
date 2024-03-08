import tensorflow as tf
import sonnet as snt
from graph_nets import modules
from graph_nets import blocks

hidden_layer_size=32    #This will be 128
output_emb_size=16      #This has to be 64 at the end
# Define the MLP model
class MLPModel_glob(snt.Module):
    def __init__(self, name=None):
        super(MLPModel_glob, self).__init__(name=name)
        self.layer1 = snt.Linear(output_size=hidden_layer_size, name='layer1')
        self.layer2 = snt.Linear(output_size=hidden_layer_size, name='layer2')
       

    def __call__(self, inputs):
        x = tf.nn.relu(self.layer1(inputs))
        out = tf.nn.relu(self.layer2(x))
        
        return out
class MLPModel_enc(snt.Module):
    def __init__(self, name=None):
        super(MLPModel_enc, self).__init__(name=name)
        self.layer1 = snt.Linear(output_size=hidden_layer_size, name='layer1')
        self.layer2 = snt.Linear(output_size=hidden_layer_size, name='layer2')
        self.layer3 = snt.Linear(output_size=hidden_layer_size, name='layer3')

        self.layer4 = snt.Linear(output_size=output_emb_size, name='layer4')

    def __call__(self, inputs):
        x = tf.nn.relu(self.layer1(inputs))
        x = tf.nn.relu(self.layer2(x))
        x = tf.nn.relu(self.layer3(x))
        out=tf.nn.relu(self.layer4(x))
        return out

# Define the Encoder layer
class Encoder(modules.GraphNetwork):
    def __init__(self):
        super(Encoder, self).__init__(
            edge_model_fn=MLPModel_enc,
            node_model_fn=MLPModel_enc,
            global_model_fn=MLPModel_glob
        )

    def __call__(self, inputs):
        return super(Encoder, self).__call__(inputs)
class MLPModel_proc(snt.Module):
    def __init__(self, name=None):
        super(MLPModel_proc, self).__init__(name=name)
        self.layer1 = snt.Linear(output_size=hidden_layer_size, name='layer1')
        self.layer2 = snt.Linear(output_size=hidden_layer_size, name='layer2')
        self.layer3 = snt.Linear(output_size=hidden_layer_size, name='layer3')
        self.layer4 = snt.Linear(output_size=output_emb_size, name='layer4')

    def __call__(self, inputs):
        x = tf.nn.relu(self.layer1(inputs))
        x = tf.nn.relu(self.layer2(x))
        x = tf.nn.relu(self.layer3(x))
        out=tf.nn.relu(self.layer4(x))
        return out
##The following class is useful to implement the residual connection as intended in the paper 
class ResidualGraphNetwork(modules.GraphNetwork):
    def __init__(self, edge_model_fn, node_model_fn, global_model_fn):
        super(ResidualGraphNetwork, self).__init__(
            edge_model_fn=MLPModel_proc(),
            node_model_fn=MLPModel_proc(),
            global_model_fn=MLPModel_glob())

    def __call__(self, input_graph):
        output_graph = super(ResidualGraphNetwork, self).__call__(input_graph)
        return output_graph.replace(
            nodes=input_graph.nodes + output_graph.nodes,
            edges=input_graph.edges + output_graph.edges,
            globals=input_graph.globals + output_graph.globals)    
class ProcessorDifferentWeights(ResidualGraphNetwork):
    def __init__(self, num_processing_steps):
        super(ProcessorDifferentWeights, self).__init__(
            edge_model_fn=lambda: MLPModel_proc(),
            node_model_fn=lambda: MLPModel_proc(),
            global_model_fn=lambda: MLPModel_glob())
        self.num_processing_steps = num_processing_steps

    def __call__(self, input_op):
        for step in range(self.num_processing_steps):
            input_op = super(ProcessorDifferentWeights, self).__call__(input_op)
        return input_op

class ProcessorSharedWeights(ResidualGraphNetwork):
    def __init__(self, num_processing_steps):
        super(ProcessorSharedWeights, self).__init__(
            edge_model_fn=MLPModel_proc,
            node_model_fn=MLPModel_proc,
            global_model_fn=MLPModel_glob)
        self.num_processing_steps = num_processing_steps

    def __call__(self, input_op):
        for step in range(self.num_processing_steps):
            input_op = super(ProcessorSharedWeights, self).__call__(input_op)
        return input_op
class GNN_double_output(snt.Module):
    def __init__(self):
        super(GNN_double_output, self).__init__()
        self.encoder = Encoder()
        self.processor = Processor(3) #Modification here change the number of layers of the processor, you can also choose another Processor from the range available
        self.decoder = Decoder()
        self.pooling_layer = PoolingLayer_double()

    def __call__(self, inputs):
        encoded = self.encoder(inputs)

        output = self.pooling_layer(encoded)
        return output