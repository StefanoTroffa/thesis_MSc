from dataclasses import dataclass, field
import tensorflow as tf
import psutil
import numpy as np 
import networkx as nx 
import sonnet as snt
import os
import time
import datetime
import gc

from compgraph.tensorflow_version.hamiltonian_operations import stochastic_gradients_tfv3, stochastic_energy_tf
from compgraph.tensorflow_version.logging_tf import log_gradient_norms, setup_tensorboard_logging
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure

from compgraph.monte_carlo import MCMCSampler,compute_phi_terms
from compgraph.useful import copy_to_non_trainable, sites_to_sparse_updated, create_amplitude_frequencies_from_graph_tuples, sparse_list_to_configs
# Set before importing TensorFlow
# import os
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_only"  # Force CPU
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Verbose logging

from compgraph.tensorflow_version.memory_control import aggressive_memory_cleanup, count_tf_objects, inspect_tf_functions

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
def log_weights_and_nan_check(step, model, writer):
    """
    Log model weight histograms and the count of NaNs.
    Software: Helps debug weight divergence or accumulation of NaNs.
    Hardware: Assists in monitoring memory usage and precision issues on GPU/CPU.
    """
    with writer.as_default():
        for var in model.trainable_variables:
            tf.summary.histogram(f"weights/{var.name}", var, step=step)
            nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32))
            tf.summary.scalar(f"nan_count/{var.name}", nan_count, step=step)
# --- Data Classes ---
@dataclass(frozen=True)
class GraphParams:
    graphType: str="2dsquare"
    n:int =2
    m: int=2
    sublattice: str = "Neel"

@dataclass(frozen=True)
class SimParams:
    beta: float = 0.05
    batch_size: int =128
    learning_rate: float= 7e-6
    outer_loop:int=60
    inner_loop:int=8

@dataclass
class Hyperams:
    symulation_type: str="VMC"
    graph_params: GraphParams=field(default_factory=GraphParams)
    sim_params: SimParams = field(default_factory=SimParams)
    ansatz: str = "GNN2simple"
    ansatz_params: dict = field(default_factory=lambda: {"hidden_size": 128, "output_emb_size": 64})



def run_tf_opt_simulation():
    energies = []
    magnetizations = []
    overlap_in_time = []
        
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
    lowest_eigenstate_as_sparse = initialize_hamiltonian_and_groundstate(
        hyperparams.graph_params,
        np.array([[int(x) for x in format(i, f'0{n_sites}b')] for i in range(2**n_sites)]) * 2 - 1
    ) if n_sites < 17 else None
    # Here we need to initialize two models otherwise tensorflow keeps the same one in the record and overwrites the properties
    model_w = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params)
    model_fix = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params)
    GT_Batch_init=initialize_graph_tuples_tf_opt(hyperparams.sim_params.batch_size, graph, subl)
    optimizer=snt.optimizers.Adam(hyperparams.sim_params.learning_rate)
    senders, receivers, edge_pairs=precompute_graph_structure(graph)
    model_w(GT_Batch_init)
    sampler_var=MCMCSampler(model_w, GT_Batch_init, hyperparams.sim_params.beta, edge_pairs=edge_pairs)
    sampler_te=MCMCSampler(model_fix, GT_Batch_init, hyperparams.sim_params.beta, graph=graph, edge_pairs=edge_pairs)
    sampler_te.model(GT_Batch_init)
    print(sampler_te.model.trainable_variables[0])

    # ======================================
    # Initialization of tensorboard logging
    # ======================================

    summary_writer, log_dir = setup_tensorboard_logging()

    print("\nTensorBoard logs written to:", log_dir)
    print("To view the results, run:")
    print(f"tensorboard --logdir {log_dir}")

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)
    GT_Batch_update=GT_Batch_init
    del GT_Batch_init
    with summary_writer.as_default():
        tf.summary.text('configuration/hyperparameters', f'beta: {hyperparams.sim_params.beta}\nlearning_rate: {hyperparams.sim_params.learning_rate}\n', step=0)

        for step in range(hyperparams.sim_params.outer_loop):
            if step==0:
                metrics = {
                'energy': None,  # Will be updated after inner loop.
                'magnetization': None,
                'overlap': None,
                'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'notes': f"Outer step {step} completed"
            }
            
            outer_start_time = time.time()  # [Integration] Start timing outer iteration.


            if physical_devices:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                metrics.update({
                    'gpu_memory_mb': gpu_memory['current'] / (1024 * 1024),
                })

            if step > 0:
                log_training_metrics(summary_writer, step, metrics)
            if step > 0:
                    with summary_writer.as_default():
                        tf.summary.scalar('training/energy', metrics['energy'] if metrics['energy'] is not None else 0, step=step)
                        tf.summary.scalar('memory/ram_used_mb', metrics['ram_used_mb'], step=step)
                        if 'gpu_memory_mb' in metrics:
                            tf.summary.scalar('memory/gpu_used_mb', metrics['gpu_memory_mb'], step=step)
                        tf.summary.text('notes/custom_message', metrics['notes'], step=step)
                    # [Integration] Log model weights and NaN counts after each outer iteration.
                    # log_weights_and_nan_check(step, sampler_var.model, summary_writer)
            
            
            # =============================
            # Updating the sampler_te.model
            # =============================
            copy_to_non_trainable(sampler_var.model, sampler_te.model)

            # tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)
 
            for innerstep in range(hyperparams.sim_params.inner_loop):           
                
                inner_start_time = time.time()  # [Integration] Start timing inner iteration.

                GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batch(GT_Batch_update, 20)
                phi_terms=compute_phi_terms(GT_Batch_update, sampler_te)

                energy, loc_energies=stochastic_energy_tf(psi_new, sampler_var, edge_pairs, GT_Batch_update, 0.0)
                stoch_loss, stoch_grads=stochastic_gradients_tfv3(phi_terms, GT_Batch_update, sampler_var)

                optimizer.apply(stoch_grads, sampler_var.model.trainable_variables)  


                spins=GT_Batch_update.nodes[:,0]
                avg_spin=tf.reduce_mean(spins)    
                print(f"innerstep {innerstep} done, energy is {energy}, average spin is{avg_spin}")
                inner_end_time = time.time()  # [Integration] End timing inner iteration.
                with summary_writer.as_default():
                    tf.summary.scalar('timing/inner_step_duration', inner_end_time - inner_start_time,
                                    step=step * hyperparams.sim_params.inner_loop + innerstep)
                energies.append(energy.numpy())
                magnetizations.append(avg_spin.numpy())
                overlap_in_time.append(tf.reduce_mean(tf.abs(phi_terms)).numpy())
                # Clean up temporary variables to help with memory management.
                metrics = {
                    'energy': energies[-1] if energies else None,
                    'magnetization': magnetizations[-1] if magnetizations else None,
                    'overlap': overlap_in_time[-1] if overlap_in_time else None,
                    'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                    'notes': f"Outer step {step} completed"
                }  
                print(f"\n=== Iteration {innerstep+1} Summary ===")
                print(metrics['ram_used_mb'])
                if innerstep==0:
                    log_gradient_norms(step * hyperparams.sim_params.inner_loop + innerstep, stoch_grads, summary_writer)
                    log_training_metrics(summary_writer, step, metrics)    
                print(metrics['ram_used_mb'])
            
                del stoch_loss, stoch_grads, energy, loc_energies, spins, avg_spin                
                del psi_new, phi_terms
        summary_writer.flush()

        outer_end_time = time.time()  # [Integration] End timing outer iteration.
        with summary_writer.as_default():
            tf.summary.scalar('timing/outer_step_duration', outer_end_time - outer_start_time, step=step)
        aggressive_memory_cleanup()
        tf.keras.backend.clear_session()
        gc.collect()
    log_weights_and_nan_check(hyperparams.sim_params.outer_loop, sampler_var.model, summary_writer)
    print("Training completed. Check TensorBoard for detailed logs.")        
    return




# --- Main Execution ---
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    # tf.config.run_functions_eagerly(True)
    run_tf_opt_simulation()    
    # Enable XLA debugging options
    # os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_only"
    # tf.debugging.enable_check_numerics()
    #===========================================
