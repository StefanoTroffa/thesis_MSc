from compgraph.tensorflow_version.graph_tuple_manipulation import create_hamiltonian_batch_jit_r1, get_single_graph_from_batch


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


@tf.function(jit_compile=True)
def graph_hamiltonian_jit(graph_tuple, edge_pairs, j2):
    # tf.print("Inputs:", tf.shape(graph_tuple.nodes), tf.shape(edge_pairs))

    config = tf.ensure_shape(graph_tuple.nodes[:,0], [None])

    # Compute Hamiltonian terms
    all_configs, all_amplitudes = config_hamiltonian_product_jit_o3(config, edge_pairs, j2)
    new_graphs = create_hamiltonian_batch_jit_r1(graph_tuple, all_configs)

    return new_graphs, all_amplitudes


# @tf.function(jit_compile=True)
# @tf.function(jit_compile=True)
@tf.function()
def stochastic_gradients_tfv3(phi_terms, GT_Batch_update, sampler_var):
        with tf.GradientTape() as tape:
            psi = sampler_var.model(GT_Batch_update)
            psi_coeff=tf.complex(
                 real=psi[:, 0] * tf.cos(psi[:, 1]),
                 imag=psi[:, 0] * tf.sin(psi[:, 1]))
            # Compute the loss function
            log_psi_conj = tf.math.log(tf.math.conj(psi_coeff))
            ratio_phi_psi = tf.stop_gradient(phi_terms / psi_coeff)
            # print(ratio_phi_psi)
            stoch_loss= tf.reduce_mean(log_psi_conj) - tf.reduce_mean(ratio_phi_psi*log_psi_conj)/tf.reduce_mean(ratio_phi_psi)
            gradients = tape.gradient(tf.math.real(stoch_loss), sampler_var.model.trainable_variables)
        del tape, psi, psi_coeff, log_psi_conj, ratio_phi_psi
        # del psi_coeff, log_psi_conj, ratio_phi_psi
        return stoch_loss,  [tf.identity(g) if g is not None else None for g in gradients]


@tf.function()
def stochastic_energy_tf(psi_new,sampler_var, edge_pairs, GT_Batch,J2):

    batch_size = tf.shape(GT_Batch.n_node)[0]
    psi_coeff=tf.complex(
            psi_new[:,0] * tf.cos( psi_new[:,1] ),
            psi_new[:,0]  * tf.sin( psi_new[:,1] ))
    def compute_local_energy(gt,i):
        """
        Compute the local energy for a single graph in the batch.
        """
        # 1. Generate all Hamiltonian configurations/amplitudes for this graph
        new_graphs, ham_amplitudes = graph_hamiltonian_jit(gt, edge_pairs, J2)

        # 2. Evaluate ψ(s) for original configuration
        # psi_s = sampler_var.evaluate_model(gt)  # Complex scalar
        psi_s=psi_coeff[i]
        # print(psi_s, psi_sv2)
        # 3. Batch evaluate ψ(s') for all H-induced configurations
        model_outputs = sampler_var.model(new_graphs)  # [num_configs, 2]
        amplitudes = model_outputs[:, 0]
        phases = model_outputs[:, 1]
        psi_s_prime = tf.complex(
            amplitudes * tf.cos(phases),
            amplitudes * tf.sin(phases))

        # 4. Compute local energy: Σ (ham_coeff * ψ(s')/ψ(s)) 
        ratios = psi_s_prime / psi_s
        ham_amplitudes = tf.cast(ham_amplitudes, tf.complex64)
        # del psi_s, psi_s_prime
        return tf.reduce_sum(ham_amplitudes * ratios)
    def single_graph_energy(i):
        single_graph = get_single_graph_from_batch(GT_Batch, i)
        return compute_local_energy(single_graph,i)
    local_energies = tf.map_fn(
        single_graph_energy,
            tf.range(batch_size),
        fn_output_signature=tf.complex64
    )
    del psi_coeff
    return tf.reduce_mean(local_energies), local_energies