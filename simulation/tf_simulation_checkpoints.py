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
from compgraph.tensorflow_version.hamiltonian_operations import stochastic_gradients_tfv3,stochastic_overlap_gradient,stochastic_energy_tf, compute_staggered_magnetization
from compgraph.tensorflow_version.logging_tf import log_gradient_norms, setup_tensorboard_loggingv2, initialize_checkpoint, log_training_metrics, log_weights_and_nan_check      
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure
from compgraph.monte_carlo import MCMCSampler,compute_phi_terms
from compgraph.useful import copy_to_non_trainable
from compgraph.tensorflow_version.model_loading import check_and_reinitialize_model

from compgraph.tensorflow_version.memory_control import aggressive_memory_cleanup, count_tf_objects, inspect_tf_functions

# --- Data Classes ---
@dataclass(frozen=True)
class GraphParams:
    graphType: str="2dsquare"
    n:int =3
    m: int=3
    # sublattice: str = "Disordered"
    sublattice: str = "Alternatepattern"
    # sublattice: str ="Neel"

@dataclass(frozen=True)
class SimParams:
    beta: float = 0.08
    batch_size: int = 256
    learning_rate: float= 2e-4
    outer_loop:int=1269
    inner_loop:int=20
    gradient:str='overlap'
    seed=860432
    # seed that works on 3x3
    # seed=860432
    # seed that works on 4x4
    # seed=860432

@dataclass
class Hyperams:
    simulation_type: str="VMC"
    # symulation_type: str="VMC2spins"    
    # symulation_type: str="ExactSim"
    graph_params: GraphParams=field(default_factory=GraphParams)
    sim_params: SimParams = field(default_factory=SimParams)
    # ansatz: str = "GNN2adv"
    ansatz: str = "GNNprocnorm"
    ansatz_params: dict = field(default_factory=lambda: {"hidden_size": 128, "output_emb_size": 64, 'K_layer':1})
    
    # ansatz: str = "GNN2simple"
    # ansatz_params: dict = field(default_factory=lambda: {"hidden_size": 128, "output_emb_size": 64})
def batch_staggered_metrics_pm1(spins_pm1, eps_pm1):
    """
    spins_pm1:  (B, N) tf.float32, values ±1
    eps_pm1:    (N,)   tf.float32, values ±1
    Returns dict with m_rms, m_abs, S_pp
    """
    spins = tf.convert_to_tensor(spins_pm1, tf.float32)
    eps   = tf.convert_to_tensor(eps_pm1,   tf.float32)
    N     = tf.cast(tf.shape(spins)[1], tf.float32)
    Ms    = tf.reduce_sum(spins * eps[None,:], axis=1)
    Ms2   = tf.reduce_mean(tf.square(Ms))      
    m_rms = tf.sqrt(Ms2) / N
    m_abs = tf.reduce_mean(tf.abs(Ms)) / N
    S_pp  = Ms2 / N
    return m_rms, m_abs, S_pp

def run_tf_opt_simulation():
    energies = []
    magnetizations = []
    overlap_in_time = []
    std_energies = []
    # =====================================================
    # Initialization of hyperparameters, graph, and models.
    # =====================================================
    hyperparams = Hyperams()
    graph, subl = create_graph_from_ham(hyperparams.graph_params.graphType,
        (hyperparams.graph_params.n, hyperparams.graph_params.m),
       
        hyperparams.graph_params.sublattice
    )

    n_sites = hyperparams.graph_params.n * hyperparams.graph_params.m
    if n_sites%2==0:
        _, subl_for_stagger= create_graph_from_ham(hyperparams.graph_params.graphType,
            (hyperparams.graph_params.n, hyperparams.graph_params.m),
            "Neel")    
    if n_sites%2==0:
        lowest_eigenstate_as_sparse = initialize_hamiltonian_and_groundstate(
            hyperparams.graph_params,
            np.array([[int(x) for x in format(i, f'0{n_sites}b')] for i in range(2**n_sites)]) * 2 - 1
        ) if n_sites < 17 else None
    else: 
        lowest_eigenstate_as_sparse=None
    model_w = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params, hyperparams.sim_params.seed)
    model_fix = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params, hyperparams.sim_params.seed)
    GT_Batch_init=initialize_graph_tuples_tf_opt(hyperparams.sim_params.batch_size, graph, subl)
    model_w(GT_Batch_init)
    tollerance_param=0.2
    model_w,final_seed=check_and_reinitialize_model(model_w, GT_Batch_init, hyperparams, tolerance=tollerance_param, max_attempts=500, seed=hyperparams.sim_params.seed)
    seed_to_save=final_seed 
    optimizer=snt.optimizers.Adam(hyperparams.sim_params.learning_rate,0.9,0.99)
    senders, receivers, edge_pairs=precompute_graph_structure(graph)
    model_fix(GT_Batch_init)
    sampler_var=MCMCSampler(GT_Batch_init, hyperparams.sim_params.beta, edge_pairs=edge_pairs)
    template_graphs_output=initialize_graph_tuples_tf_opt(tf.shape(edge_pairs)[0]+1,graph,subl)
    sampler_te=MCMCSampler(GT_Batch_init, template=template_graphs_output, 
                           beta=hyperparams.sim_params.beta, edge_pairs=edge_pairs)    
    # sampler_te.model(GT_Batch_init)

    # ======================================
    # Initialization of tensorboard logging
    # ======================================

    summary_writer, log_dir = setup_tensorboard_loggingv2(hyperparams,'checkpointed_logs')
    tf.print("\nTensorBoard logs written to:", log_dir)
    print("To view the results, run:")
    print(f"tensorboard --logdir {log_dir}")
    physical_devices = tf.config.list_physical_devices('GPU')
    time_start= time.time()  # [Integration] Start timing the entire simulation.
    GT_Batch_update=GT_Batch_init
    thermalization_steps=500*n_sites
    for i in range(thermalization_steps):
       if hyperparams.simulation_type=="VMC2spins":
            GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batchv2(model_w,GT_Batch_update)
       elif hyperparams.simulation_type=="VMC":
            # GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batch(model_w,GT_Batch_update)

           GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batch_profilemem(model_w,GT_Batch_update)
    tf.print(f"Thermalization completed in {time.time()-time_start:.2f} seconds.")
    del GT_Batch_init

    #=========================================
    template_graphs_output=initialize_graph_tuples_tf_opt(tf.shape(edge_pairs)[0]+1,graph,subl)
    if n_sites%2==0:  
        subl_idx = tf.argmax(subl_for_stagger, axis=1)  # shape (N,)
        stagger_factor_single = tf.where(subl_idx == 0, 1.0, -1.0)
        stagger_factor_batch = tf.tile(stagger_factor_single, multiples=[hyperparams.sim_params.batch_size])
        stagger_2d = tf.reshape(stagger_factor_batch, (hyperparams.sim_params.batch_size, n_sites))
    else:
        stagger_factor_single = tf.ones((n_sites,), dtype=tf.float32)
        stagger_factor_batch = tf.tile(stagger_factor_single, multiples=[hyperparams.sim_params.batch_size])
        stagger_2d = tf.reshape(stagger_factor_batch, (hyperparams.sim_params.batch_size, n_sites))
    # =======================================
    ## We are now saving the seed value of model initialization as a text in tensorflow events. 
    with summary_writer.as_default():
        tf.summary.text('configuration/hyperparameters', 
                    f'beta: {hyperparams.sim_params.beta}\nlearning_rate: {hyperparams.sim_params.learning_rate}\nseed: {seed_to_save}\n',

                    step=0)

        for step in range(hyperparams.sim_params.outer_loop):
            # Initial setup for step 0
            if step == 0:
                checkpoint_manager = initialize_checkpoint(log_dir, model_w, optimizer)
                log_weights_and_nan_check(step, model_w, summary_writer)
            
            # Track timing and metrics for outer loop
            outer_start_time = time.time()
            
            # Memory tracking
            memory_metrics = {
                'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024)
            }
            if physical_devices:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                memory_metrics['gpu_memory_mb'] = gpu_memory['peak'] / (1024 * 1024)
            
            # Checkpointing and memory cleanup
            if step % 30 == 0 and step > 0:
                # aggressive_memory_cleanup()
                # tf.keras.backend.clear_session()
                gc.collect()
                log_weights_and_nan_check(step, model_w, summary_writer)
                checkpoint_path = checkpoint_manager.save()
                print(f"Model checkpoint saved at step {step} to: {checkpoint_path}")
            
            # Update target model
            copy_to_non_trainable(model_w, model_fix)
            
            for innerstep in range(hyperparams.sim_params.inner_loop):
                # Track all timings
                timing_metrics = {}
                global_step = step * hyperparams.sim_params.inner_loop + innerstep
                inner_start_time = time.time()
                
                # Monte Carlo updates
                mc_start = time.time()
                for i in range(2*n_sites):
                    if hyperparams.simulation_type == "VMC2spins":
                        GT_Batch_update, psi_new = sampler_var.monte_carlo_update_on_batchv2(model_w, GT_Batch_update)
                    elif hyperparams.simulation_type == "VMC":
                        # GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batch(model_w,GT_Batch_update)

                        GT_Batch_update, psi_new = sampler_var.monte_carlo_update_on_batch_profilemem(model_w, GT_Batch_update)
                timing_metrics['mc_duration'] = time.time() - mc_start
                
                # Phi terms computation
                phi_start = time.time()
                phi_terms = compute_phi_terms(GT_Batch_update, sampler_te, model_fix)
                timing_metrics['phi_terms_duration'] = time.time() - phi_start
                
                # Energy calculation
                energy_start = time.time()
                energy, std_energy, loc_energies = stochastic_energy_tf(
                    psi_new, model_w, edge_pairs, template_graphs_output, GT_Batch_update, 0.0)
                timing_metrics['energy_duration'] = time.time() - energy_start
                
                # Gradient computation and optimization
                grad_start = time.time()
                # stoch_loss, stoch_grads, stoch_overlap, grad_norms = improved_stochastic_gradients(
                #     phi_terms, GT_Batch_update, model_w)
                stoch_loss, stoch_grads= stochastic_gradients_tfv3(
                    phi_terms, GT_Batch_update, model_w)
                optimizer.apply(stoch_grads, model_w.trainable_variables)
                timing_metrics['gradient_duration'] = time.time() - grad_start
                
                # Calculate magnetization
                spins = GT_Batch_update.nodes[:, 0]

                spins_2d = tf.reshape(spins, (hyperparams.sim_params.batch_size, n_sites))
                m_rms, m_abs, S_pp=batch_staggered_metrics_pm1(
                    spins_2d, stagger_factor_single)
                avg_spin = tf.reduce_mean(spins)
                timing_metrics['inner_step_duration'] = time.time() - inner_start_time
                avg_staggered_abs_magnetization = compute_staggered_magnetization(spins_2d, stagger_2d,n_sites)
                # Collect training metrics
                training_metrics = {
                    'energy_real': tf.math.real(energy),
                    'energy_imag': tf.math.imag(energy),
                    'std_energy': std_energy/tf.math.sqrt(tf.cast(loc_energies.shape[0], tf.float32)),
                    'magnetization': avg_spin,
                    'staggered_magnetization_abs': avg_staggered_abs_magnetization,
                    'staggered_magnetization_sqrt': m_rms,
                    'staggered_magnetization_absv2': m_abs,
                    'staggered_magnetization_S_pp': S_pp,
                }
                
                # Store metrics for outer loop logging
                energies.append(tf.math.real(energy).numpy())
                std_energies.append(std_energy.numpy())
                magnetizations.append(avg_spin.numpy())
                
                # Write all metrics to TensorBoard at once
                with summary_writer.as_default():
                    # Write timing metrics
                    for metric_name, value in timing_metrics.items():
                        tf.summary.scalar(f'timing/{metric_name}', value, step=global_step)
                    
                    # Write training metrics
                    for metric_name, value in training_metrics.items():
                        tf.summary.scalar(f'training/{metric_name}', value, step=global_step)
                    
                    # Only write memory metrics on first inner step
                    if innerstep == 0:
                        for metric_name, value in memory_metrics.items():
                            tf.summary.scalar(f'memory/{metric_name}', value, step=step)
                        
                        # Write custom message
                        tf.summary.text('notes/custom_message', 
                                    f"Outer step {step} completed", step=step)
                     # Log gradient norms periodically
                    if (step % 30 == 0 and innerstep == 0):
                        log_gradient_norms(global_step, stoch_grads, summary_writer)
                # Clean up temporary variables
                del stoch_loss, stoch_grads, energy, loc_energies, spins, avg_spin, avg_staggered_abs_magnetization
                del psi_new, phi_terms
            
            # Track outer loop duration
            outer_duration = time.time() - outer_start_time
            with summary_writer.as_default():
                tf.summary.scalar('timing/outer_step_duration', outer_duration, step=step)
            
           
            
            # Flush summary writer occasionally
            if step % 50 == 0:
                summary_writer.flush()
            
        log_weights_and_nan_check(hyperparams.sim_params.outer_loop, model_w, summary_writer)
        print("Training completed. Check TensorBoard for detailed logs.")        
        return




# --- Main Execution ---
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    # tf.config.run_functions_eagerly(True)
    # with tf.device('/GPU:0'):
    run_tf_opt_simulation()    
    # Enable XLA debugging options
    # os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_only"
    # tf.debugging.enable_check_numerics()
    #===========================================
