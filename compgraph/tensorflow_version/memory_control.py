import tensorflow as tf


import gc


def aggressive_memory_cleanup():
    """Aggressive memory cleanup to free TensorFlow resources"""
    # Clear Python objects
    gc.collect()

    # Clear TensorFlow session
    tf.keras.backend.clear_session()

    # Force TensorFlow to release memory back to the system
    if tf.config.list_physical_devices('GPU'):
        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            print(f"Before GPU cleanup: {gpu_memory['current'] / 1024 / 1024:.1f} MB")
        except:
            pass

        # Reset GPU memory stats
        tf.config.experimental.reset_memory_stats('GPU:0')

        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            print(f"After GPU cleanup: {gpu_memory['current'] / 1024 / 1024:.1f} MB")
        except:
            pass


def count_tf_objects():
    """Count TensorFlow objects in memory"""
    tensors = [obj for obj in gc.get_objects()
               if isinstance(obj, tf.Tensor) or isinstance(obj, tf.Variable)]
    return len(tensors)


def inspect_tf_functions():
    """Inspect compiled TensorFlow functions"""
    concrete_functions = []
    for obj in gc.get_objects():
        if isinstance(obj, tf.types.experimental.ConcreteFunction):
            concrete_functions.append(obj)
    return len(concrete_functions)