import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from scipy.sparse import coo_matrix
import numpy as np
from compgraph.cg_repr import graph_tuple_to_config_hamiltonian_product, square_2dham_exp, config_hamiltonian_product
from compgraph.useful import graph_tuple_toconfig, sparse_list_to_configs
from compgraph.useful import sites_to_sparse

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


#TODO deprecated, eliminate this function and substitute all recurrences with v2
def variational_wave_function_on_batch(model, graph_batch, graph_batch_indices):
    unique_data = {}  # Dictionary to store unique indices and their corresponding values
    size=2**len(graph_batch[0].nodes)
    # Compute the wave function components for each graph tuple
    for idx, graph_tuple in enumerate(graph_batch):
        #print(graph_batch_indices, type(graph_batch_indices))
        # Extract the row index from the configuration
        row_index = graph_batch_indices[idx].indices[0]
        # Check if the index is already in the dictionary
        if row_index in unique_data:
            pass  # Sum up the values for repeated indices
        else:
            amplitude, phase = model(graph_tuple)[0]
            # Convert amplitude to complex tensor with zero imaginary part
            complex_coefficient=tf.complex(real=amplitude, imag=phase)

        
            unique_data[row_index] = complex_coefficient  # Add new index and value to the dictionary
    
    # Convert dictionary to lists
    values = list(unique_data.values())
    indices = [[key, 0] for key in unique_data.keys()]
    
    # Convert lists to tensors
    values_tensor = tf.stack(values, axis=0)
    indices_tensor = tf.constant(indices, dtype=tf.int64)
    
    # Create a sparse tensor
    sparse_tensor = tf.sparse.SparseTensor(indices=indices_tensor, values=values_tensor, dense_shape=[size, 1])
    
    return tf.sparse.reorder(sparse_tensor)

def time_evoluted_config_amplitude(model, beta, graph_tuple, graph, sublattice_encoding):
    graph_tuples_nonzero, amplitudes_gt=graph_tuple_to_config_hamiltonian_product(graph_tuple, graph, sublattice_encoding)
    final_amplitude=[]
    for i, gt in enumerate(graph_tuples_nonzero):
        amplitude, phase = model(gt)[0]
        amplitude *= amplitudes_gt[i]
        complex_coefficient=tf.complex(real=amplitude, imag=phase)

        final_amplitude.append(complex_coefficient)
    beta= -1.*beta
    total_amplitude = tf.multiply(beta,tf.reduce_sum(tf.stack(final_amplitude)))
    amplitude, phase = model(graph_tuple)[0]
    complex_coefficient=tf.complex(real=amplitude, imag=phase)

    total_amplitude = tf.add(complex_coefficient, total_amplitude)
    return total_amplitude

def time_evoluted_wave_function_on_batch(model_te, beta, graph_batch,graph, sublattice_encoding):
    unique_data = {}  # Dictionary to store unique indices and their corresponding values
    size=2**len(graph_batch[0].nodes)
    # Compute the wave function components for each graph tuple
    for graph_tuple in graph_batch:
        #print(graph_batch_indices, type(graph_batch_indices))
        # Extract the row index from the configuration
        config=graph_tuple_toconfig(graph_tuple)
        sparse_not= sites_to_sparse([config])[0][0]
        row_index = sparse_not.indices[0]
        # Check if the index is already in the dictionary
        if row_index in unique_data:
            pass  
        else:
            complex_coefficient=time_evoluted_config_amplitude(model_te, beta, graph_tuple, graph, sublattice_encoding)

        
            unique_data[row_index] = complex_coefficient  # Add new index and value to the dictionary
    
    # Convert dictionary to lists
    values = list(unique_data.values())
    indices = [[key, 0] for key in unique_data.keys()]
    
    # Convert lists to tensors
    values_tensor = tf.stack(values, axis=0)
    indices_tensor = tf.constant(indices, dtype=tf.int64)
    
    # Create a sparse tensor
    sparse_tensor = tf.sparse.SparseTensor(indices=indices_tensor, values=values_tensor, dense_shape=[size, 1])
 
    
    return tf.sparse.reorder(sparse_tensor)
def variational_wave_function_on_batch_v2(model, graph_batch):
    unique_data = {}  # Dictionary to store unique indices and their corresponding values
    size=2**len(graph_batch[0].nodes)
    # Compute the wave function components for each graph tuple
    for graph_tuple in graph_batch:
        #print(graph_batch_indices, type(graph_batch_indices))
        # Extract the row index from the configuration
        config=graph_tuple_toconfig(graph_tuple)
        sparse_not= sites_to_sparse([config])[0][0]
        row_index = sparse_not.indices[0]        # Check if the index is already in the dictionary
        if row_index in unique_data:
            pass  # Sum up the values for repeated indices
        else:
            amplitude, phase = model(graph_tuple)[0]
            # Convert amplitude to complex tensor with zero imaginary part
            complex_coefficient=tf.complex(real=amplitude, imag=phase)

        
            unique_data[row_index] = complex_coefficient  # Add new index and value to the dictionary
    
    # Convert dictionary to lists
    values = list(unique_data.values())
    indices = [[key, 0] for key in unique_data.keys()]
    
    # Convert lists to tensors
    values_tensor = tf.stack(values, axis=0)
    indices_tensor = tf.constant(indices, dtype=tf.int64)
    
    # Create a sparse tensor
    sparse_tensor = tf.sparse.SparseTensor(indices=indices_tensor, values=values_tensor, dense_shape=[size, 1])
    
    return tf.sparse.reorder(sparse_tensor)

def sparse_tensor_exp_energy(wave_function, graph, J2):
    #wave_conj= tf.sparse.map_values(tf.math.conj, wave_function)
    wave_conj=tf.sparse.map_values(tf.math.conj, wave_function)
    ket=np.array(wave_function.values)
    bra=np.array(wave_conj.values)

    bra_indices=np.array(wave_function.indices)[:, 0]
    num_sites=len(graph.nodes)
    bra_configs=sparse_list_to_configs(bra_indices,num_sites)
    print(bra_configs)
    ket_configs=bra_configs
    exp_value=0.
    
    for idx_bra, config_bra in enumerate(bra_configs):
        configurations_nonzero, coefficients = config_hamiltonian_product(config_bra, graph)
        for idx_ket,config_ket in enumerate(ket_configs):
            match_indices = np.where(np.all(configurations_nonzero == config_ket, axis=1))[0]
            if match_indices.size > 0:
                idx_nonzero = match_indices[0]  # Assuming the first match's index if multiple matches
                coefficient = coefficients[idx_nonzero]  # Get the corresponding coefficient
                exp_value += ket[idx_ket] * bra[idx_bra] * coefficient
        
    return exp_value
    


def montecarlo_overlap_time_evoluted(te_coefficients, graph_tuples_te, model_var, graph_tuples_var):
    expectation_over_te_distribution=0.
    expectation_over_var_distribution=0.
    coefficients_var_on_var=variational_wave_function_on_batch


    pass
def evolving_function(wave, Ham_tensor,beta):
    wave=tf.sparse.reorder(wave)
    auxphi= adjust_dtype_and_multiply(Ham_tensor,wave)
    beta *= -1
    phi=tf.sparse.map_values(tf.multiply,auxphi, beta)
    #print(wave.indices,phi.indices)
    
    phi=tf.sparse.add(wave,tf.stop_gradient(phi))
    #print(phi)
    wave_with_0=tf.sparse.map_values(tf.multiply,phi, 0)
    wave=tf.sparse.add(wave,wave_with_0)
    psi_conj= tf.sparse.map_values(tf.math.conj, wave)
    overlap=tf.sparse.map_values(tf.multiply,psi_conj,phi)
    norm_wave = tf.norm(wave.values)
    norm_ito_wave=tf.norm(phi.values)
    
    normalization=1/tf.math.sqrt(norm_wave*norm_ito_wave)
    #print(normalization, 'This is the coefficient at denominator')
    overlap_normalized=tf.sparse.map_values(tf.multiply,overlap,normalization)
    return -overlap_normalized.values
