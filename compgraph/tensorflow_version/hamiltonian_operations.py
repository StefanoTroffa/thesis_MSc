from compgraph.tensorflow_version.graph_tuple_manipulation import update_graph_tuples_config_jit, create_hamiltonian_batch_jit_r1, get_single_graph_from_batch, create_hamiltonian_batch_xla


import tensorflow as tf


def config_hamiltonian_product_jit_o3(config, edge_pairs, j2: float = 0.0):
    """
    Vectorized JIT-compilable version of the Hamiltonian application.

    Args:
      config: A 1D tensor (shape [num_nodes]) representing the current spin configuration.
      edge_pairs: A 2D tensor of shape [num_edges, 2] containing pairs of node indices (edges).
      j2: Next-nearest neighbor parameter (not used here; kept for interface consistency).

    Returns:
      A tuple (new_configs, new_amplitudes) where:
         new_configs is a tensor of candidate configurations (each row is one candidate configuration)
         new_amplitudes is a 1D tensor of the corresponding amplitudes.

    The function implements the following logic:
      1. For each edge (i, j):
         - If config[i] equals config[j], add +0.25 to a diagonal accumulator.
         - Otherwise, subtract 0.25.
      2. For edges where config[i] != config[j], produce a candidate configuration by flipping 
         the spins at i and j (i.e. candidate[i] = -config[i] and candidate[j] = -config[j]).
         The amplitude for such an update is multiplier*0.5.
      3. If the total diagonal contribution is nonzero, append the original configuration with 
         amplitude = multiplier*(diagonal_total).
    """
    # Ensure config is a tensor (its dtype should be compatible with equality comparisons)
    config = tf.convert_to_tensor(config)
    num_edges = tf.shape(edge_pairs)[0]

    # Gather spins for each edge
    spin_i = tf.gather(config, edge_pairs[:, 0])
    spin_j = tf.gather(config, edge_pairs[:, 1])

    # Compute per-edge diagonal contributions: +0.25 if spins equal, -0.25 if different.
    diag_contrib = tf.where(tf.equal(spin_i, spin_j), 0.25, -0.25)
    diagonal_total = tf.reduce_sum(diag_contrib)

    # For each edge, we want to create a candidate config by flipping the spins at the two nodes.
    # Define a function that given an edge returns the candidate configuration.
    def candidate_fn(edge):
        # edge is a vector of shape [2] containing indices [i, j].
        # Create a copy of the original config.
        new_config = tf.identity(config)
        # Flip the spins at positions i and j.
        # (Flipping here means multiplying by -1.)
        indices = tf.reshape(edge, [2, 1])
        updates = -tf.gather(new_config, edge)
        new_config = tf.tensor_scatter_nd_update(new_config, indices, updates)
        return new_config

    # Apply candidate_fn to each edge.
    candidate_configs_all = tf.map_fn(candidate_fn, edge_pairs, dtype=config.dtype)

    # Only keep candidates where the spins differ (i.e. where an off-diagonal update is allowed).
    mask = tf.not_equal(spin_i, spin_j)
    candidate_configs = tf.boolean_mask(candidate_configs_all, mask)

    # Determine the multiplier (if the number of nodes is 4, multiplier=2, else 1).
    multiplier = tf.cond(
        tf.equal(tf.shape(config)[0], 4),
        lambda: tf.constant(2.0, dtype=tf.float32),
        lambda: tf.constant(1.0, dtype=tf.float32)
    )

    # Off-diagonal amplitudes: each candidate gets multiplier * 0.5.
    off_diag_amplitudes = multiplier * 0.5 * tf.ones(tf.shape(candidate_configs)[0], dtype=tf.float32)

    # If the diagonal_total is nonzero, append the original configuration with amplitude multiplier * diagonal_total.
    def add_diag():
        new_configs = tf.concat([candidate_configs, tf.expand_dims(config, 0)], axis=0)
        new_amplitudes = tf.concat([off_diag_amplitudes, [multiplier * diagonal_total]], axis=0)
        return new_configs, new_amplitudes

    def no_diag():
        return candidate_configs, off_diag_amplitudes

    new_configs, new_amplitudes = tf.cond(tf.not_equal(diagonal_total, 0.0), add_diag, no_diag)
    print("output shape of new configs and amplitudes", tf.shape(new_configs),tf.shape(new_amplitudes))
    return new_configs, new_amplitudes

def config_hamiltonian_product_xla_improved(config, edge_pairs, j2: float = 0.0):
    """XLA-compatible implementation with fixed config size."""
    config = tf.convert_to_tensor(config)
    num_edges = tf.shape(edge_pairs)[0]
    
    # Get the configuration size (number of nodes)
    # Try to get static shape first, fall back to dynamic if needed
    config_size = config.shape[0]
    if config_size is None:  # If shape is dynamic
        config_size = tf.shape(config)[0]
    
    # Gather spins for each edge
    spin_i = tf.gather(config, edge_pairs[:, 0])
    spin_j = tf.gather(config, edge_pairs[:, 1])
    
    # Compute diagonal contributions
    spin_equal = tf.cast(tf.equal(spin_i, spin_j), tf.float32)
    diag_contrib = 0.25 * (2.0 * spin_equal - 1.0)
    diagonal_total = tf.reduce_sum(diag_contrib)
    
    # Create validity mask (1.0 where spins differ, 0.0 where equal)
    valid_edge_mask = 1.0 - spin_equal  # 1.0 where spins differ
    
    # Use TensorArray with fixed size
    config_array = tf.TensorArray(
        dtype=config.dtype,
        size=num_edges,
        dynamic_size=False,
        clear_after_read=False
    )
    
    # Loop through edges and create configurations
    for i in tf.range(num_edges):
        new_config = tf.identity(config)
        idx_i = edge_pairs[i, 0]
        idx_j = edge_pairs[i, 1]
        
        # Apply both flips in a single operation if possible
        indices = tf.stack([idx_i, idx_j])
        updates = -tf.gather(new_config, indices)
        new_config = tf.tensor_scatter_nd_update(
            new_config,
            tf.reshape(indices, [2, 1]),
            updates
        )
        
        # Store in the array
        config_array = config_array.write(i, new_config)
    
    # Stack configurations
    all_configs_tensor = config_array.stack()
    
    # Determine multiplier
    multiplier = tf.cond(
        tf.equal(config_size, 4),
        lambda: tf.constant(2.0, dtype=tf.float32),
        lambda: tf.constant(1.0, dtype=tf.float32)
    )
    
    # Create amplitudes: 0.5*multiplier where valid, 0.0 where invalid
    edge_amplitudes = multiplier * 0.5 * valid_edge_mask
    
    # Always add the original configuration with its diagonal amplitude
    all_configs_with_orig = tf.concat([all_configs_tensor, tf.expand_dims(config, 0)], axis=0)
    all_amplitudes = tf.concat([edge_amplitudes, [multiplier * diagonal_total]], axis=0)
    
    return all_configs_with_orig, all_amplitudes

@tf.function(jit_compile=True)
def graph_hamiltonian_jit(graph_tuple, edge_pairs, j2):
    # tf.print("Inputs:", tf.shape(graph_tuple.nodes), tf.shape(edge_pairs))

    config = tf.ensure_shape(graph_tuple.nodes[:,0], [None])

    # Compute Hamiltonian terms
    all_configs, all_amplitudes = config_hamiltonian_product_jit_o3(config, edge_pairs, j2)
    new_graphs = create_hamiltonian_batch_jit_r1(graph_tuple, all_configs)

    return new_graphs, all_amplitudes

@tf.function(jit_compile=True)
def graph_hamiltonian_jit_xla(graph_tuple, edge_pairs, j2, template_graph):
    # tf.print("Inputs:", tf.shape(graph_tuple.nodes), tf.shape(edge_pairs))

    # config = tf.ensure_shape(graph_tuple.nodes[:,0], [None])

    # Compute Hamiltonian terms
    all_configs, all_amplitudes = config_hamiltonian_product_xla_improved(graph_tuple.nodes[:,0], edge_pairs, j2)
    new_graphs = update_graph_tuples_config_jit(template_graph, all_configs)

    return new_graphs, all_amplitudes

@tf.function()
def stochastic_gradients_tfv3(phi_terms, GT_Batch_update, model):
        with tf.GradientTape() as tape:
            psi = model(GT_Batch_update)
            psi_coeff=tf.complex(
                 real=psi[:, 0] * tf.cos(psi[:, 1]),
                 imag=psi[:, 0] * tf.sin(psi[:, 1]))
            # Compute the loss function
            log_psi_conj = tf.math.log(tf.math.conj(psi_coeff))
            ratio_phi_psi = tf.stop_gradient(phi_terms / psi_coeff)
            # print(ratio_phi_psi)
            stoch_loss= tf.reduce_mean(log_psi_conj) - tf.reduce_mean(ratio_phi_psi*log_psi_conj)/tf.reduce_mean(ratio_phi_psi)
            gradients = tape.gradient(tf.math.real(stoch_loss), model.trainable_variables)
        del tape, psi, psi_coeff, log_psi_conj, ratio_phi_psi
        # del psi_coeff, log_psi_conj, ratio_phi_psi

        return stoch_loss,gradients

# @tf.function()
def stochastic_overlap_gradient(phi_terms, GT_Batch, sampler_var):
    """
    Correctly implements the gradient of log overlap with respect to model parameters,
    using the gradient of log(psi).
    """
    with tf.GradientTape() as tape:
        tape.watch(sampler_var.model.trainable_variables)
        psi = sampler_var.model(GT_Batch)
        psi_coeff = tf.complex(
            psi[:,0] * tf.cos(psi[:,1]),
            psi[:,0] * tf.sin(psi[:,1]))
        log_psi = tf.math.log(psi_coeff)
    
    log_psi_gradients = tape.gradient(log_psi, sampler_var.model.trainable_variables)
    phi_psi_ratio = phi_terms / psi_coeff
    
    avg_phi_psi = tf.reduce_mean(phi_psi_ratio)
    
    final_gradients = []
    print("log psi gradients", tf.shape(log_psi_gradients))
    print("phi psi ratio", tf.shape(phi_psi_ratio))
    print("avg phi psi", tf.shape(avg_phi_psi)) 
    for g in log_psi_gradients:
        if g is None:
            final_gradients.append(None)
        else:
            final_gradients.append(tf.math.real(tf.reduce_mean(phi_psi_ratio * g) / avg_phi_psi)
            )

    weighted_term= tf.reduce_mean(phi_psi_ratio * log_psi_gradients) / avg_phi_psi
    unweighted_term = tf.reduce_mean(log_psi_gradients)
    final_gradients = tf.math.real(weighted_term - unweighted_term)
   
    del tape, psi, psi_coeff, log_psi_gradients, phi_psi_ratio, avg_phi_psi
 
    return [tf.identity(g) if g is not None else None for g in final_gradients]

@tf.function()
def stochastic_energy_tf(psi_new,model_var, edge_pairs,template_graph, GT_Batch,J2):

    batch_size = tf.shape(GT_Batch.n_node)[0]
    psi_coeff=tf.complex(
            psi_new[:,0] * tf.cos( psi_new[:,1] ),
            psi_new[:,0]  * tf.sin( psi_new[:,1] ))
    def compute_local_energy(gt,i,model_var):
        """
        Compute the local energy for a single graph in the batch:
        1. Generate all Hamiltonian configurations/amplitudes for this graph.
        2. Evaluate ψ(s) for original configuration.
        3. Batch evaluate ψ(s') for all H-induced configurations.
        4. Compute local energy: Σ (ham_coeff * ψ(s')/ψ(s)) 
        5. Return the local energy.
        """

        new_graphs, ham_amplitudes = graph_hamiltonian_jit_xla(gt, edge_pairs, J2, template_graph)

        psi_s=psi_coeff[i]

        model_outputs = model_var(new_graphs)  
        amplitudes = model_outputs[:, 0]
        phases = model_outputs[:, 1]
        psi_s_prime = tf.complex(
            amplitudes * tf.cos(phases),
            amplitudes * tf.sin(phases))

        ratios = psi_s_prime / psi_s
        ham_amplitudes = tf.cast(ham_amplitudes, tf.complex64)
        # del psi_s, psi_s_prime
        return tf.reduce_sum(ham_amplitudes * ratios)
    def single_graph_energy(i):
        single_graph = get_single_graph_from_batch(GT_Batch, i)
        return compute_local_energy(single_graph,i,model_var)
    local_energies = tf.map_fn(
        single_graph_energy,
            tf.range(batch_size),
        fn_output_signature=tf.complex64
    )
    del psi_coeff
    mean_energy=tf.reduce_mean(local_energies)
    energy_std=tf.sqrt(tf.reduce_mean(tf.abs(local_energies - mean_energy)**2))    
    return mean_energy, energy_std, local_energies



def compute_staggered_magnetization(spins, stagger_factor, num_sites):
    """
    Compute the staggered magnetization for a batch of graphs.
    
    Parameters:
    -----------
    spins : tf.Tensor
        Tensor of shape (B, N) containing spin configurations.
    stagger_factor : tf.Tensor
        Tensor of shape (B, N) containing the staggered factor for each spin.
    
    Returns:
    --------
    Mstag : tf.Tensor
        Staggered magnetization for each graph in the batch.
    """
    # Compute the staggered magnetization
    staggered_2d = spins * stagger_factor
    Mstag_each_graph = tf.reduce_sum(staggered_2d, axis=1) / tf.cast(num_sites, tf.float32)

    # take absolute value => shape (B,)
    abs_each_graph = tf.abs(Mstag_each_graph)

    # if you want the overall average across the batch
    abs_Mstag_mean = tf.reduce_mean(abs_each_graph)
    
    return abs_Mstag_mean