from scipy.sparse import csr_matrix, eye
import tensorflow as tf
def compute_wave_function_csr(graph_tuples_batch, ansatz, configurations):
    #TO BE FIXED the wave function currently sums up same configurations coefficients if they are presented multiple times. What should I do here? 
    
    data = []  # List to store the non-zero entries
    row_indices = []  # List to store the row indices
    col_indices = [0] * len(configurations)  # Column indices are all 0 for a vector
    size= 2**len(graph_tuples_batch[0].nodes)

    # Compute the wave function components for each graph tuple
    for idx, graph_tuple in enumerate(graph_tuples_batch):
        amplitude, phase = ansatz(graph_tuple)[0]
        amplitude = tf.cast(amplitude, tf.float32)
        phase = tf.cast(phase, tf.float32)
        
        # Now create the complex coefficient
        complex_coefficient = tf.complex(amplitude, phase)
        # Compute the complex coefficient using TensorFlow operations
        #print(complex_coefficient,configurations[idx])
        
        # Extract the row index from the configuration
        row_index = configurations[idx].indices[0]
        
        # Append the data and indices to the lists
        data.append(complex_coefficient.numpy())
        row_indices.append(row_index)
    
    # Construct the csr_matrix using the accumulated data and indices 
    #Shape it is tbd by other means
    wave_function_csr = csr_matrix((data, (row_indices, col_indices)), shape=(size, 1))
    
    return wave_function_csr
def innerprod_sparse(psi_sparse, phi_sparse):
    inn_prod=psi_sparse.conj().transpose().dot(phi_sparse)   
    return inn_prod
def construct_sparse_hamiltonian(Graph, spin_operators, J2):
    Hamiltonian = csr_matrix((2**Graph.number_of_nodes(), 2**Graph.number_of_nodes()))
    # Define the coupling constants
    J1 = 1. 
    node_to_index=nd_to_index(Graph)

    # Iterate over the edges of the graph
    for i, j in Graph.edges:
        print(i,j)
        # Map nodes to indices
        i_index = node_to_index[i]
        j_index = node_to_index[j]

        # Add the interaction term to the Hamiltonian
        term_to_be_add = (0.5*spin_operators[0][i_index]*(spin_operators[1][j_index]) + 
                          0.5*spin_operators[1][i_index]*(spin_operators[0][j_index]) + 
                          spin_operators[2][i_index]*(spin_operators[2][j_index]))
        Hamiltonian += J1 * csr_matrix(term_to_be_add)

    # Add next nearest neighbour interactions
    for i in Graph.nodes:
        i_index = node_to_index[i]
        for j in Graph.neighbors(i):
            j_index = node_to_index[j]
            for k in Graph.neighbors(j):
                if k != i:
                    k_index = node_to_index[k]
                    term= csr_matrix(0.5*spin_operators[0][i_index]*spin_operators[1][k_index] 
                                     + 0.5*spin_operators[1][i_index]*spin_operators[0][k_index] 
                                     +spin_operators[2][i_index]*spin_operators[2][k_index])

                    Hamiltonian += J2 * term
    return Hamiltonian                
def loss_sparse_vectors(psi_sparse, phi_sparse):
    psi_norm= innerprod_sparse(psi_sparse,psi_sparse)
    phi_norm= innerprod_sparse(phi_sparse,phi_sparse)
    norm_sqrt = tf.math.sqrt(psi_norm[0,0] * phi_norm[0,0])
    
    numerator = innerprod_sparse(psi_sparse, phi_sparse)
    loss= tf.constant(1.0, dtype=tf.float32)-numerator/norm_sqrt
    return loss