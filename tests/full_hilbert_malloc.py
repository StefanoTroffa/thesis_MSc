import tensorflow as tf
import psutil
import numpy as np 
import networkx as nx 
import sonnet as snt
import os
import time
import datetime
import gc

from dataclasses import dataclass, field

# Import required modules
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure
from compgraph.tensorflow_version.logging_tf import log_gradient_norms, setup_tensorboard_logging, log_weights_and_nan_check
from compgraph.tensorflow_version.memory_control import aggressive_memory_cleanup, count_tf_objects, inspect_tf_functions
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams, initialize_hamiltonian_and_groundstate
from compgraph.useful import copy_to_non_trainable
# --- Data Classes ---
@dataclass(frozen=True)
class GraphParams:
    graphType: str = "2dsquare"
    n: int = 2
    m: int = 2
    sublattice: str = "Disordered"

@dataclass(frozen=True)
class SimParams:
    beta: float = 0.05
    learning_rate: float = 7e-3
    outer_loop: int = 60
    inner_loop: int = 8

@dataclass
class Hyperams:
    symulation_type: str = "ExactVMC"
    graph_params: GraphParams = field(default_factory=GraphParams)
    sim_params: SimParams = field(default_factory=SimParams)
    ansatz: str = "GNN2simple"
    ansatz_params: dict = field(default_factory=lambda: {"hidden_size": 128, "output_emb_size": 32})


def log_training_metrics(summary_writer, step, metrics_dict):
    """Log training metrics to TensorBoard"""
    with summary_writer.as_default():
        tf.summary.scalar('training/energy', metrics_dict['energy'], step=step)
        if 'magnetization' in metrics_dict:
            tf.summary.scalar('training/magnetization', metrics_dict['magnetization'], step=step)
        if 'overlap' in metrics_dict:
            tf.summary.scalar('training/overlap', metrics_dict['overlap'], step=step)
        tf.summary.scalar('memory/ram_used_mb', metrics_dict['ram_used_mb'], step=step)
        if 'gpu_memory_mb' in metrics_dict:
            tf.summary.scalar('memory/gpu_used_mb', metrics_dict['gpu_memory_mb'], step=step)
        if 'learning_rate' in metrics_dict:
            tf.summary.scalar('training/learning_rate', metrics_dict['learning_rate'], step=step)
        if 'notes' in metrics_dict:
            tf.summary.text('notes/custom_message', metrics_dict['notes'], step=step)


def compute_exact_energy(model, graph_tuples, edge_pairs, J2=0.0):
    """
    Compute the exact energy by evaluating all configurations in the Hilbert space.
    
    Args:
        model: The neural network model
        graph_tuples: The full set of graph tuples representing all states
        edge_pairs: Edge pairs in the graph
        J2: Next-nearest neighbor interaction strength
        
    Returns:
        The exact energy as a tensor
    """
    batch_size = tf.shape(graph_tuples.n_node)[0]
    
    # Evaluate the model on all states
    outputs = model(graph_tuples)
    amplitudes = outputs[:, 0]
    phases = outputs[:, 1]
    
    # Convert to complex coefficients
    psi_coeffs = tf.complex(
        real=amplitudes * tf.cos(phases),
        imag=amplitudes * tf.sin(phases)
    )
    
    # Compute the probability distribution
    psi_squared = tf.abs(psi_coeffs)**2
    norm = tf.reduce_sum(psi_squared)
    probabilities = psi_squared / norm
    
    # Initialize energy accumulator
    total_energy = tf.constant(0.0, dtype=tf.float32)
    
    # Function to compute energy for each configuration
    def compute_config_energy(i):
        # Get the graph tuple for this configuration
        single_graph = graph_tuples_slice(graph_tuples, i)
        
        # Compute all possible Hamiltonian operations (off-diagonal terms)
        new_graphs, ham_amplitudes = graph_hamiltonian_jit(single_graph, edge_pairs, J2)
        
        # Evaluate all potential new states
        new_outputs = model(new_graphs)
        new_amplitudes = new_outputs[:, 0]
        new_phases = new_outputs[:, 1]
        new_psi_coeffs = tf.complex(
            real=new_amplitudes * tf.cos(new_phases),
            imag=new_amplitudes * tf.sin(new_phases)
        )
        
        # Compute ratios ψ(s')/ψ(s)
        ratios = new_psi_coeffs / psi_coeffs[i]
        
        # Compute local energy for this configuration
        return tf.reduce_sum(tf.cast(ham_amplitudes, tf.complex64) * ratios)
    
    # Map the function across all configurations
    local_energies = tf.map_fn(
        compute_config_energy,
        tf.range(batch_size),
        fn_output_signature=tf.complex64
    )
    
    # Weight the energy by probability
    weighted_energy = tf.reduce_sum(tf.cast(probabilities, tf.complex64) * local_energies)
    
    return tf.math.real(weighted_energy)

from compgraph.tensorflow_version.hamiltonian_operations import graph_hamiltonian_jit
def compute_exact_overlap(model_var, model_te, graph_tuples, beta, edge_pairs, J2=0.0):
    """
    Compute the exact overlap between variational and time-evolved states
    
    Args:
        model_var: The variational model
        model_te: The time evolution model
        graph_tuples: The full set of graph tuples
        beta: The imaginary time parameter
        edge_pairs: Edge pairs in the graph
        J2: Next-nearest neighbor interaction strength
        
    Returns:
        The overlap between the two states
    """
    batch_size = tf.shape(graph_tuples.n_node)[0]
    
    # Evaluate both models on all states
    var_outputs = model_var(graph_tuples)
    var_amplitudes = var_outputs[:, 0]
    var_phases = var_outputs[:, 1]
    var_psi = tf.complex(
        real=var_amplitudes * tf.cos(var_phases),
        imag=var_amplitudes * tf.sin(var_phases)
    )
    
    # For time-evolved state, we need to apply (1-βH) to each state
    # First, get the base psi values
    te_outputs = model_te(graph_tuples)
    te_amplitudes = te_outputs[:, 0]
    te_phases = te_outputs[:, 1]
    te_psi_base = tf.complex(
        real=te_amplitudes * tf.cos(te_phases),
        imag=te_amplitudes * tf.sin(te_phases)
    )
    
    # Function to compute time-evolved coefficient for a single configuration
    def compute_te_coeff(i):
        # Get single graph tuple
        single_graph = graph_tuples_slice(graph_tuples, i)
        
        # Compute Hamiltonian operations
        new_graphs, ham_amplitudes = graph_hamiltonian_jit(single_graph, edge_pairs, J2)
        
        # Evaluate model on new graphs
        new_outputs = model_te(new_graphs)
        new_amplitudes = new_outputs[:, 0]
        new_phases = new_outputs[:, 1]
        new_psi = tf.complex(
            real=new_amplitudes * tf.cos(new_phases),
            imag=new_amplitudes * tf.sin(new_phases)
        )
        
        # Apply H to psi: H|ψ⟩ = ∑_j h_ij |j⟩
        h_psi = tf.reduce_sum(tf.cast(ham_amplitudes, tf.complex64) * new_psi)
        
        # Apply (1-βH) to psi: (1-βH)|ψ⟩ = |ψ⟩ - β(H|ψ⟩)
        return te_psi_base[i] - tf.cast(beta, tf.complex64) * h_psi
    
    # Map the function across all configurations
    te_psi = tf.map_fn(
        compute_te_coeff,
        tf.range(batch_size),
        fn_output_signature=tf.complex64
    )
    
    # Normalize both states
    var_norm = tf.sqrt(tf.reduce_sum(tf.abs(var_psi)**2))
    te_norm = tf.sqrt(tf.reduce_sum(tf.abs(te_psi)**2))
    
    var_psi_norm = var_psi / var_norm
    te_psi_norm = te_psi / te_norm
    
    # Compute overlap: |⟨ψ_var|ψ_te⟩|^2
    overlap = tf.abs(tf.reduce_sum(tf.math.conj(var_psi_norm) * te_psi_norm))**2
    
    return overlap


@tf.function
def exact_gradients(model_var, model_te, graph_tuples, beta, edge_pairs, J2=0.0):
    """
    Compute the exact gradients for the variational model
    
    Args:
        model_var: The variational model
        model_te: The time evolution model
        graph_tuples: The full set of graph tuples
        beta: The imaginary time parameter
        edge_pairs: Edge pairs in the graph
        J2: Next-nearest neighbor interaction strength
        
    Returns:
        The gradients for the variational model parameters
    """
    with tf.GradientTape() as tape:
        # Compute the loss as the negative logarithm of the overlap
        overlap = compute_exact_overlap(model_var, model_te, graph_tuples, beta, edge_pairs, J2)
        loss = -tf.math.log(overlap)
    
    # Compute gradients
    gradients = tape.gradient(loss, model_var.trainable_variables)
    
    return loss, gradients


def graph_tuples_slice(graph_tuples, idx):
    """Utility function to extract a single graph from a batch"""
    from compgraph.tensorflow_version.graph_tuple_manipulation import get_single_graph_from_batch
    return get_single_graph_from_batch(graph_tuples, idx)


def run_exact_hilbert_simulation():
    """
    Run a simulation using the exact full Hilbert space approach.
    """
    energies = []
    magnetizations = []
    overlaps = []
        
    # ===============================
    # Initialization of hyperparameters, graph, and models.
    # ===============================
    hyperparams = Hyperams()
    graph, subl = create_graph_from_ham(
        hyperparams.graph_params.graphType,
        (hyperparams.graph_params.n, hyperparams.graph_params.m),
        hyperparams.graph_params.sublattice
    )

    n_sites = hyperparams.graph_params.n * hyperparams.graph_params.m
    
    # Only run for small systems (n_sites <= 10)
    if n_sites > 10:
        print(f"System size too large for exact approach: {n_sites} sites would require 2^{n_sites} = {2**n_sites} states")
        print("Please use a system with 10 or fewer sites.")
        return
    
    print(f"Running exact Hilbert space simulation for {n_sites} sites (2^{n_sites} = {2**n_sites} states)")
    
    # Get the ground state if system is small enough
    lowest_eigenstate_as_sparse = initialize_hamiltonian_and_groundstate(
        hyperparams.graph_params,
        np.array([[int(x) for x in format(i, f'0{n_sites}b')] for i in range(2**n_sites)]) * 2 - 1
    ) if n_sites < 17 else None
    
    # Initialize models
    model_var = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params)
    model_te = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params)
    
    # Generate the full Hilbert space
    full_hilbert_tuples = initialize_graph_tuples_tf_opt(1, graph, subl, full_size_hilbert='yes')
    
    # Initialize optimizer
    optimizer = snt.optimizers.Adam(hyperparams.sim_params.learning_rate)
    
    # Precompute graph structure
    senders, receivers, edge_pairs = precompute_graph_structure(graph)
    
    # Initialize models with a sample input
    model_var(full_hilbert_tuples)
    model_te(full_hilbert_tuples)
    print("Models initialized")
    
    # ======================================
    # Initialization of tensorboard logging
    # ======================================
    summary_writer, log_dir = setup_tensorboard_logging('fh_run')
    print("\nTensorBoard logs written to:", log_dir)
    print("To view the results, run:")
    print(f"tensorboard --logdir {log_dir}")

    # Check for GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)
    
    # Import necessary functions from hamiltonian_operations
    from compgraph.tensorflow_version.hamiltonian_operations import graph_hamiltonian_jit
    
    with summary_writer.as_default():
        tf.summary.text('configuration/hyperparameters', 
                       f'Exact Hilbert space simulation\n'
                       f'beta: {hyperparams.sim_params.beta}\n'
                       f'learning_rate: {hyperparams.sim_params.learning_rate}\n'
                       f'n_sites: {n_sites}\n'
                       f'hilbert_dimension: {2**n_sites}', step=0)
        
        for step in range(hyperparams.sim_params.outer_loop):
            metrics = {
                'energy': 0.0 if not energies else energies[-1],
                'magnetization': 0.0 if not magnetizations else magnetizations[-1],
                'overlap': 0.0 if not overlaps else overlaps[-1],
                'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'notes': f"Outer step {step} starting"
            }
            
            if physical_devices:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                metrics.update({
                    'gpu_memory_mb': gpu_memory['current'] / (1024 * 1024),
                })
            
            if step > 0:
                log_training_metrics(summary_writer, step, metrics)
            
            outer_start_time = time.time()
            
            # # Copy weights from var model to te model
            # for var_param, te_param in zip(model_var.trainable_variables, model_te.trainable_variables):
            #     te_param.assign(var_param)
            copy_to_non_trainable(model_var, model_te)

            # Inner training loop
            for innerstep in range(hyperparams.sim_params.inner_loop):
                inner_start_time = time.time()
                
                # Compute loss and gradients
                loss, gradients = exact_gradients(
                    model_var, model_te, full_hilbert_tuples, 
                    hyperparams.sim_params.beta, edge_pairs
                )
                
                # Apply gradients
                optimizer.apply(gradients, model_var.trainable_variables)
                
                # Compute energy
                energy = compute_exact_energy(model_var, full_hilbert_tuples, edge_pairs)
                
                # Compute magnetization
                # Extract spins from all configurations
                configs = full_hilbert_tuples.nodes[:, 0]
                configs_reshaped = tf.reshape(configs, [2**n_sites, n_sites])
                
                # Evaluate model on all configurations
                outputs = model_var(full_hilbert_tuples)
                amplitudes = outputs[:, 0]
                phases = outputs[:, 1]
                psi = tf.complex(
                    real=amplitudes * tf.cos(phases),
                    imag=amplitudes * tf.sin(phases)
                )
                
                # Compute probabilities
                probs = tf.abs(psi)**2
                probs = probs / tf.reduce_sum(probs)
                
                # Compute magnetization for each configuration
                config_mags = tf.reduce_sum(configs_reshaped, axis=1) / n_sites
                
                # Weight by probabilities
                magnetization = tf.reduce_sum(config_mags * tf.cast(probs, config_mags.dtype))
                
                # Compute overlap with exact ground state if available
                if lowest_eigenstate_as_sparse is not None:
                    # Normalize the variational state
                    psi_normalized = psi / tf.norm(psi)
                    
                    # Reshape to match sparse tensor format
                    psi_sparse = tf.sparse.from_dense(tf.reshape(psi_normalized, [2**n_sites, 1]))
                    
                    # Compute overlap
                    overlap = tf.abs(tf.sparse.sparse_dense_matmul(
                        tf.sparse.transpose(lowest_eigenstate_as_sparse),
                        tf.sparse.to_dense(psi_sparse)
                    ))[0, 0]
                else:
                    overlap = tf.constant(0.0)
                
                # Log metrics
                energies.append(energy.numpy())
                magnetizations.append(magnetization.numpy())
                overlaps.append(overlap.numpy())
                
                # Print progress
                print(f"Iteration {step}.{innerstep}: Energy = {energy.numpy():.6f}, "
                      f"Magnetization = {magnetization.numpy():.6f}, "
                      f"Overlap = {overlap.numpy():.6f}, "
                      f"Loss = {loss.numpy():.6f}")
                
                inner_end_time = time.time()
                with summary_writer.as_default():
                    tf.summary.scalar('timing/inner_step_duration', 
                                     inner_end_time - inner_start_time,
                                     step=step * hyperparams.sim_params.inner_loop + innerstep)
                
                # Log gradient norms
                if innerstep == 0:
                    log_gradient_norms(
                        step * hyperparams.sim_params.inner_loop + innerstep, 
                        gradients, 
                        summary_writer
                    )
            
            # Update metrics after inner loop
            metrics = {
                'energy': energies[-1],
                'magnetization': magnetizations[-1],
                'overlap': overlaps[-1],
                'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'notes': f"Outer step {step} completed"
            }
            
            if physical_devices:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                metrics.update({
                    'gpu_memory_mb': gpu_memory['current'] / (1024 * 1024),
                })
            
            log_training_metrics(summary_writer, step, metrics)
            
            # Log weights and check for NaNs
            log_weights_and_nan_check(step, model_var, summary_writer)
            
            outer_end_time = time.time()
            with summary_writer.as_default():
                tf.summary.scalar('timing/outer_step_duration', 
                                 outer_end_time - outer_start_time, 
                                 step=step)
            
            # Clean up memory
            aggressive_memory_cleanup()
        
        # Final logging
        log_weights_and_nan_check(hyperparams.sim_params.outer_loop, model_var, summary_writer)
        
    print("Training completed. Check TensorBoard for detailed logs.")
    return energies, magnetizations, overlaps


if __name__ == "__main__":
    # Attempt to enable XLA JIT compilation for better performance
    # try:
    #     tf.config.optimizer.set_jit(True)
    #     print("XLA JIT compilation enabled.")
    # except:
    #     print("Could not enable XLA JIT compilation.")
    
    # Run the simulation
    run_exact_hilbert_simulation()