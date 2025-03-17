import tensorflow as tf
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
from scipy.sparse import coo_matrix
import numpy as np
from compgraph.cg_repr import graph_tuple_to_config_hamiltonian_product_update, square_2dham_exp, config_hamiltonian_product 
from compgraph.useful import graph_tuple_toconfig, sparse_list_to_configs, graph_tuple_list_to_configs_list, sites_to_sparse, sites_to_sparse_updated
# import line_profiler
# import atexit
# profile2 = line_profiler.LineProfiler()
# atexit.register(profile2.print_stats)
def convert_csr_to_sparse_tensor(csr_matrix):
    coo = coo_matrix(csr_matrix)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data.astype(np.complex128), coo.shape)

def create_sparsetensor_from_configs_amplitudes(configurations, amplitudes, num_sites):
    """
    configurations is supposed to be a nd.array where the first axis iterates through different configurations
    amplitudes is as well an nd.array with complex entries where
    configurations[0] corresponds to amplitudes[0]
    """
    # Create sparse vector using TensorFlow
    # sparse_indices = sites_to_sparse(configurations)[0] 
    # indices = [[idx.indices[0], 0] for idx in sparse_indices]  # Format indices for tf.sparseTensor
    indices = [[idx, 0] for idx in sites_to_sparse_updated(configurations)]  # Format indices for tf.sparseTensor

    values_tensor = tf.stack(amplitudes, axis=0)
    indices_tensor = tf.constant(indices, dtype=tf.int64)
    sparse_tensor = tf.sparse.SparseTensor(indices=indices_tensor, values=values_tensor, dense_shape=[2**num_sites, 1])
    return tf.sparse.reorder(sparse_tensor)

def create_sparse_tensor_from_graph_tuples_amplitudes(graph_tuples, amplitudes):
    n_sites=len(graph_tuples[0].nodes)
    configurations= graph_tuple_list_to_configs_list(graph_tuples)
    sparse_tensor=create_sparsetensor_from_configs_amplitudes(configurations, amplitudes, n_sites)
    return tf.sparse.reorder(sparse_tensor)


# @profile2
def time_evoluted_config_amplitude(model, beta, graph_tuple, graph):
    graph_tuples_nonzero, amplitudes_gt=graph_tuple_to_config_hamiltonian_product_update(graph_tuple, graph)
    final_amplitude=[]
    for i, gt in enumerate(graph_tuples_nonzero):
        amplitude, phase = model(gt)[0]
        amplitude *= amplitudes_gt[i]
        complex_coefficient= tf.complex(real=amplitude* tf.cos(phase), imag=amplitude * tf.sin(phase))

        final_amplitude.append(complex_coefficient)
    beta= -1.*beta
    total_amplitude = tf.multiply(beta,tf.reduce_sum(tf.stack(final_amplitude)))
    complex_coefficient=evaluate_model(model, graph_tuple)

    total_amplitude = tf.add(complex_coefficient, total_amplitude)
    return total_amplitude

# @profile2
def graph_tuple_to_row_index(graph_tuple):
    config=graph_tuple_toconfig(graph_tuple)
    sparse_not= sites_to_sparse_updated([config])[0]
    # sparse_not= sites_to_sparse([config])[0][0]

    # row_index = sparse_not.indices[0]
    return sparse_not

def time_evoluted_wave_function_on_batch(model_te, beta, graph_batch,graph):
    unique_data = {}  # Dictionary to store unique indices and their corresponding values
    size=2**len(graph_batch[0].nodes)
    # Compute the wave function components for each graph tuple
    for graph_tuple in graph_batch:
        #print(graph_batch_indices, type(graph_batch_indices))
        # Extract the row index from the configuration
        row_index= graph_tuple_to_row_index(graph_tuple)
        # Check if the index is already in the dictionary
        if row_index in unique_data:
            #unique_data[row_index] += time_evoluted_config_amplitude(model_te, beta, graph_tuple, graph, sublattice_encoding)
            
            pass # previously there was no row above, and then we'd just ignore the repeated index. This however, does not reward the MC method  
        else:
            complex_coefficient=time_evoluted_config_amplitude(model_te, beta, graph_tuple, graph)

        
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

def evaluate_model(model, graph_tuple):
    amplitude, phase = model(graph_tuple)[0]
    return  tf.complex(real=amplitude* tf.cos(phase), imag=amplitude * tf.sin(phase)) 

def variational_wave_function_on_batch(model, graph_batch):
    unique_data = {}  # Dictionary to store unique indices and their corresponding values
    size=2**len(graph_batch[0].nodes)
    # Compute the wave function components for each graph tuple
    for graph_tuple in graph_batch:
        #print(graph_batch_indices, type(graph_batch_indices))
        # Extract the row index from the configuration
        row_index = graph_tuple_to_row_index(graph_tuple)
        # Check if the index is already in the dictionary
        if row_index in unique_data:
            #unique_data[row_index] += evaluate_model(model, graph_tuple) 
            pass  # Sum up the values for repeated indices
        else:
            unique_data[row_index] = evaluate_model(model, graph_tuple)
    
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
    #print(bra_configs)
    ket_configs=bra_configs
    exp_value=0.
    normalization_factor= 1/tf.norm(wave_function.values)

    for idx_bra, config_bra in enumerate(bra_configs):
        configurations_nonzero, coefficients = config_hamiltonian_product(config_bra, graph)
        for idx_ket,config_ket in enumerate(ket_configs):
            match_indices = np.where(np.all(configurations_nonzero == config_ket, axis=1))[0]
            if match_indices.size > 0:
                idx_nonzero = match_indices[0]  # Assuming the first match's index if multiple matches
                coefficient = coefficients[idx_nonzero]  # Get the corresponding coefficient
                exp_value += ket[idx_ket] * bra[idx_bra] * coefficient
        
    return exp_value*normalization_factor**2

def calculate_sparse_overlap(left_part, right_part):
    """
    Calculate the overlap using sparse tensors for coefficients.
    This computation calculates: <psi|phi> without the square module.
    sparse_coefficients_te_on: SparseTensor of time-evolved model coefficients.
    sparse_coefficients_var_on: SparseTensor of variational model coefficients.
    """
    # Conjugate the time-evolved coefficients
    conjugated_te = tf.sparse.map_values(tf.math.conj, left_part)
    wave_left_with_0=tf.sparse.map_values(tf.multiply,conjugated_te, 0)
    wave_right_with_0= tf.sparse.map_values(tf.multiply,right_part, 0)
    wave_0=tf.sparse.add(wave_right_with_0,wave_left_with_0)
    conjugated_te=tf.sparse.add(conjugated_te,wave_0)
    right_part= tf.sparse.add(right_part,wave_0)
    # Element-wise multiplication of sparse tensors
    product_sparse = tf.sparse.map_values(tf.multiply, conjugated_te, right_part)

    # Sum all the elements to get the expectation value
    overlap = tf.sparse.reduce_sum(product_sparse)

    return overlap

def montecarlo_logloss_overlap_time_evoluted(coefficients_te_on_te, graph_tuples_te, model_var, model_te, graph_tuples_var,beta, graph_hamiltonian):
    
    """
    
    Expectation value over the time-evolved distribution:
    E[TE] = sum(|ψ_TE(β)|^2 * (ψ_var_on_TE / ψ_TE_on_TE))= sum((ψ_TE(β)*) o ψ_var_on_TE)
    Where |ψ_TE(β)|^2 represents the norm squared of the time-evolved state coefficients,
    and ψ_var_on_TE / ψ_TE_on_TE is the ratio of variational coefficients to time-evolved coefficients on the TE distribution.
    
    """
    
    overlap_over_te_distribution=0.
    overlap_over_var_distribution=0.
    coefficients_var_on_var=variational_wave_function_on_batch(model_var,graph_tuples_var)
    coefficients_var_on_te=variational_wave_function_on_batch(model_var, graph_tuples_te)    
    coefficients_te_on_var=time_evoluted_wave_function_on_batch(model_te,beta, graph_tuples_var, graph_hamiltonian)
    overlap_over_te_distribution=calculate_sparse_overlap(coefficients_te_on_te, coefficients_var_on_te)
    overlap_over_var_distribution=calculate_sparse_overlap(coefficients_var_on_var, coefficients_te_on_var)
    norm_wave = tf.norm(coefficients_var_on_var.values)
    norm_te_wave=tf.norm(coefficients_te_on_te.values)
    normalization=1/(norm_wave*norm_te_wave)**2

    overlap=tf.math.sqrt(overlap_over_te_distribution*overlap_over_var_distribution*normalization)
    #print("Overlap according to MC function", overlap_over_te_distribution*overlap_over_var_distribution*normalization)
    return -tf.math.log(overlap)


def quimb_vec_to_sparse(vector, configurations, num_sites):
    ampl = np.array(vector).flatten()
    # Check if the vector is complex
    if np.iscomplexobj(ampl):
        ampl_complex = tf.convert_to_tensor(ampl, dtype=tf.complex64)
    else:
        # If not complex, create the complex tensor
        ampl_complex = tf.complex(ampl, tf.zeros_like(ampl))
    
    # Create the sparse tensor from configurations and complex amplitudes
    sp2 = create_sparsetensor_from_configs_amplitudes(configurations, ampl_complex, num_sites)
    return  tf.sparse.reorder(sp2)
