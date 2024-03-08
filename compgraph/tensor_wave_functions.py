import tensorflow as tf
def compute_wave_function_sparse_tensor(graph_tuples_batch, ansatz, configurations):
    
    values = []  # List to store the non-zero entries
    indices = []  # List to store the indices of non-zero entries
    
    # Compute the wave function components for each graph tuple
    for idx, graph_tuple in enumerate(graph_tuples_batch):
        amplitude, phase = ansatz(graph_tuple)[0]
        
        # Convert amplitude to complex tensor with zero imaginary part
        amplitude_complex = tf.complex(real=amplitude, imag=tf.zeros_like(amplitude))
        
        # Compute the complex coefficient using TensorFlow operations
        complex_coefficient = amplitude_complex * tf.math.exp(tf.complex(real=0.0, imag=phase))
        
        # Extract the row index from the configuration
        row_index = configurations[idx].indices[0]
        
        # Append the data and indices to the lists
        values.append(complex_coefficient)
        indices.append([row_index, 0])
    
    # Convert lists to tensors
    values_tensor = tf.stack(values, axis=0)
    indices_tensor = tf.constant(indices, dtype=tf.int64)
    
    # Create a sparse tensor
    sparse_tensor = tf.sparse.SparseTensor(indices=indices_tensor, values=values_tensor, dense_shape=[512, 1])
    
    return sparse_tensor
def compute_wave_function_sparse_tensor_u2(graph_tuples_batch, ansatz, configurations):
    #### Through numerical simulation the runtime of the function that only gives unique values is approx equal to not check at all
    unique_data = {}  # Dictionary to store unique indices and their corresponding values
    size=2**len(graph_tuples_batch[0].nodes)
    # Compute the wave function components for each graph tuple
    for idx, graph_tuple in enumerate(graph_tuples_batch):
        amplitude, phase = ansatz(graph_tuple)[0]
        # Convert amplitude to complex tensor with zero imaginary part
        amplitude_complex = tf.complex(real=amplitude, imag=tf.zeros_like(amplitude))
        
        # Compute the complex coefficient using TensorFlow operations
        complex_coefficient = amplitude_complex * tf.math.exp(tf.complex(real=0.0, imag=phase))
        
        # Extract the row index from the configuration
        row_index = configurations[idx].indices[0]
        
        # Check if the index is already in the dictionary
        if row_index in unique_data:
            pass  # Sum up the values for repeated indices
        else:
            unique_data[row_index] = complex_coefficient  # Add new index and value to the dictionary
    
    # Convert dictionary to lists
    values = list(unique_data.values())
    indices = [[key, 0] for key in unique_data.keys()]
    
    # Convert lists to tensors
    values_tensor = tf.stack(values, axis=0)
    indices_tensor = tf.constant(indices, dtype=tf.int64)
    
    # Create a sparse tensor
    sparse_tensor = tf.sparse.SparseTensor(indices=indices_tensor, values=values_tensor, dense_shape=[size, 1])
    
    return sparse_tensor
