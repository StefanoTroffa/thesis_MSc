import sonnet as snt
import tensorflow as tf
def nd_to_index(Graph):
    nodes_idx = {node: i for i, node in enumerate(Graph.nodes)}
    return nodes_idx
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