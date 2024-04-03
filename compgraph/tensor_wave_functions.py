import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from scipy.sparse import coo_matrix
import numpy as np
def convert_csr_to_sparse_tensor(csr_matrix):
    coo = coo_matrix(csr_matrix)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data.astype(np.complex128), coo.shape)

def adjust_dtype_and_multiply(a: tf.SparseTensor, b: tf.SparseTensor):
    # Convert both SparseTensors to CSR sparse matrix with the desired type
    a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
        a.indices, tf.cast(a.values, tf.complex128), a.dense_shape)
    
    b_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
        b.indices, tf.cast(b.values, tf.complex128), b.dense_shape)

    # Perform sparse matrix multiplication
    c_sm = sparse_csr_matrix_ops.sparse_matrix_sparse_mat_mul(
        a_sm, b_sm, type=tf.complex128)

    # Convert the result back to a SparseTensor
    c = sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(
        c_sm, tf.complex128)

    return tf.SparseTensor(indices=c.indices, values=c.values, dense_shape=c.dense_shape)
def compute_loss_tensor(psi_sparse, phi_sparse):
    # Compute the conjugate of the sparse tensor values
    psi_sparse_conj = tf.SparseTensor(
        indices=psi_sparse.indices,
        values=tf.math.conj(psi_sparse.values),
        dense_shape=psi_sparse.dense_shape
    )
    
    phi_sparse_conj = tf.SparseTensor(
        indices=phi_sparse.indices,
        values=tf.math.conj(phi_sparse.values),
        dense_shape=phi_sparse.dense_shape
    )
    
    # Reorder the indices of the conjugate sparse tensor
    psi_sparse_conj = tf.sparse.reorder(psi_sparse_conj)
    
    # Convert psi_sparse_conj to dense
    psi_dense_conj = tf.sparse.to_dense(psi_sparse_conj)
    phi_dense_conj= tf.sparse.to_dense(phi_sparse_conj)
    
    # Compute the norms using sparse-dense matrix multiplication
    
    psi_norm = tf.sparse.sparse_dense_matmul(psi_sparse, psi_dense_conj, adjoint_b=False)
    phi_norm = tf.sparse.sparse_dense_matmul(phi_sparse, phi_dense_conj, adjoint_b=False)
    print("norms:", psi_norm, phi_norm)
    norm = psi_norm * phi_norm
    
    # Compute the numerator using dense-sparse matrix multiplication
    numerator = tf.sparse.sparse_dense_matmul(psi_sparse, phi_dense_conj, adjoint_b=True)
    print(psi_sparse, phi_sparse)
    print("numerator", numerator, "Norm" ,"\n", norm)
    loss = 1-numerator/tf.math.sqrt(norm)
    
    return loss
def compute_wave_function_sparse_tensor(graph_tuples_batch, ansatz, configurations):
    size=2**len(graph_tuples_batch[0].nodes)

    #we assume COMPLETET HERE THE THING. 
    values = []  # List to store the non-zero entries
    indices = []  # List to store the indices of non-zero entries
    # Compute the wave function components for each graph tuple
    for idx, graph_tuple in enumerate(graph_tuples_batch):
        amplitude, phase = ansatz(graph_tuple)[0]
        amplitude = tf.cast(amplitude, tf.float32)
        phase = tf.cast(phase, tf.float32)        
 
        # Compute the complex coefficient using TensorFlow operations
        complex_coefficient = tf.complex(amplitude, phase)
        
        # Extract the row index from the configuration
        row_index = configurations[idx].indices[0]
        
        # Append the data and indices to the lists
        values.append(complex_coefficient)
        indices.append([row_index, 0])
    
    # Convert lists to tensors
    values_tensor = tf.stack(values, axis=0)
    indices_tensor = tf.constant(indices, dtype=tf.int64)
    
    # Create a sparse tensor
    sparse_tensor = tf.sparse.SparseTensor(indices=indices_tensor, values=values_tensor, dense_shape=[size, 1])
    
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
