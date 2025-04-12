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
from compgraph.tensorflow_version.hamiltonian_operations import stochastic_gradients_tfv3,stochastic_overlap_gradient,stochastic_energy_tf
from compgraph.tensorflow_version.logging_tf import log_gradient_norms, setup_tensorboard_loggingv2, initialize_checkpoint
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure
from compgraph.monte_carlo import MCMCSampler,compute_phi_terms
from compgraph.useful import copy_to_non_trainable
from compgraph.tensorflow_version.logging_tf import log_training_metrics, log_weights_and_nan_check         
from compgraph.tensorflow_version.model_loading import check_and_reinitialize_model

from compgraph.tensorflow_version.memory_control import aggressive_memory_cleanup, count_tf_objects, inspect_tf_functions
@tf.function()
def improved_stochastic_gradients(phi_terms, GT_Batch_update, model):
    """
    Improved stochastic gradient implementation with better numerical stability
    and more direct calculation of the gradient of the log overlap.
    
    Args:
        phi_terms: Target state coefficients (time evolved or ground state)
        GT_Batch_update: Graph tuple batch for all configurations
        model: The variational neural network model
        
    Returns:
        loss: The loss value (negative log overlap)
        gradients: Gradients for the model parameters
    """
    with tf.GradientTape() as tape:
        # Get the neural network output
        psi = model(GT_Batch_update)
        
        # Convert output to complex coefficients
        psi_coeff = tf.complex(
            real=psi[:, 0] * tf.cos(psi[:, 1]),
            imag=psi[:, 0] * tf.sin(psi[:, 1])
        )
        
        # Normalize both wavefunctions
        psi_norm = tf.norm(psi_coeff) + 1e-12
        phi_norm = tf.norm(phi_terms) + 1e-12
        
        psi_normalized = psi_coeff / psi_norm
        phi_normalized = phi_terms / phi_norm
        # Compute overlap directly
        overlap = tf.abs(tf.reduce_sum(tf.math.conj(psi_normalized) * phi_normalized))**2
        err_bias=0
        # Loss is negative log of the overlap (we want to maximize overlap)
        loss = -tf.math.log(overlap + err_bias)
        # Calculate amplitude variance to encourage diversity
        # amplitude_variance = tf.math.reduce_variance(psi[:, 0])
        # phase_variance = tf.math.reduce_variance(psi[:, 1])
        
        # # Diversity penalty (small weight at first)
        # diversity_weight = 0.1
        # diversity_loss = -diversity_weight * (amplitude_variance + phase_variance)        
        # Add a small L2 regularization to prevent parameter explosion
        # l2_reg = 1e-5 * sum(tf.reduce_sum(tf.square(w)) for w in model.trainable_variables)
        # regularized_loss = loss + l2_reg 
        regularized_loss = loss

    # Calculate gradients
    gradients_before = tape.gradient(regularized_loss, model.trainable_variables)
    
    # Clip gradients to prevent explosions
    gradients, grad_norm = tf.clip_by_global_norm(gradients_before, 1.0)

    return loss, gradients, overlap, grad_norm
# --- Data Classes ---
@dataclass(frozen=True)
class GraphParams:
    graphType: str="2dsquare"
    n:int =4
    m: int=4
    # sublattice: str = "Disordered"
    sublattice: str ="Neel"

@dataclass(frozen=True)
class SimParams:
    beta: float = 0.2
    batch_size: int =128
    learning_rate: float= 7e-4  
    outer_loop:int=1024
    inner_loop:int=15
    gradient:str='overlap'

@dataclass
class Hyperams:
    simulation_type: str="VMC"
    # symulation_type: str="VMC2spins"    
    # symulation_type: str="ExactSim"
    graph_params: GraphParams=field(default_factory=GraphParams)
    sim_params: SimParams = field(default_factory=SimParams)
    ansatz: str = "GNN2adv"
    ansatz_params: dict = field(default_factory=lambda: {"hidden_size": 128, "output_emb_size": 64, 'K_layer': 3})



def run_tf_opt_simulation():
    energies = []
    magnetizations = []
    overlap_in_time = []
    std_energies = []
    # =====================================================
    # Initialization of hyperparameters, graph, and models.
    # =====================================================
    hyperparams = Hyperams()
    graph, subl = create_graph_from_ham(
        hyperparams.graph_params.graphType,
        (hyperparams.graph_params.n, hyperparams.graph_params.m),
        hyperparams.graph_params.sublattice
    )
    
    n_sites = hyperparams.graph_params.n * hyperparams.graph_params.m
    if n_sites%2==0:
        lowest_eigenstate_as_sparse = initialize_hamiltonian_and_groundstate(
            hyperparams.graph_params,
            np.array([[int(x) for x in format(i, f'0{n_sites}b')] for i in range(2**n_sites)]) * 2 - 1
        ) if n_sites < 17 else None
    else: 
        lowest_eigenstate_as_sparse=None
    model_w = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params)
    model_fix = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params)
    GT_Batch_init=initialize_graph_tuples_tf_opt(hyperparams.sim_params.batch_size, graph, subl, sz_sector=5)
    model_w(GT_Batch_init)
    
    model_w=check_and_reinitialize_model(model_w, GT_Batch_init, hyperparams, tolerance=0.01, max_attempts=50)
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
    print("\nTensorBoard logs written to:", log_dir)
    print("To view the results, run:")
    print(f"tensorboard --logdir {log_dir}")
    physical_devices = tf.config.list_physical_devices('GPU')
    time_start= time.time()  # [Integration] Start timing the entire simulation.
    GT_Batch_update=GT_Batch_init
    thermalization_steps=1000
    for i in range(thermalization_steps):
       if hyperparams.simulation_type=="VMC2spins":
            GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batchv2(model_w,GT_Batch_update)
       elif hyperparams.simulation_type=="VMC":
           GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batch(model_w,GT_Batch_update)
    tf.print(f"Thermalization completed in {time.time()-time_start:.2f} seconds.")
    del GT_Batch_init
    template_graphs_output=initialize_graph_tuples_tf_opt(tf.shape(edge_pairs)[0]+1,graph,subl)

    # with summary_writer.as_default():
    #     tf.summary.text('configuration/hyperparameters', f'beta: {hyperparams.sim_params.beta}\nlearning_rate: {hyperparams.sim_params.learning_rate}\n', step=0)

    #     for step in range(hyperparams.sim_params.outer_loop):
    #         if step==0:
    #             metrics = {
    #             'energy': None,  # Will be updated after inner loop.
    #             'std_energy': None,
    #             'magnetization': None,
    #             # 'overlap': None,
    #             'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
    #             'notes': f"Outer step {step} completed"
    #             }
    #             checkpoint_manager = initialize_checkpoint(log_dir, model_w, optimizer)
            
            
    #         outer_start_time = time.time()  # [Integration] Start timing outer iteration.


    #         if physical_devices:
    #             gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
    #             metrics.update({
    #                 'gpu_memory_mb': gpu_memory['peak'] / (1024 * 1024),
    #             })

    #         if step > 0:
    #             log_training_metrics(summary_writer, step, metrics)
    #         if step > 0:
    #                 with summary_writer.as_default():
    #                     tf.summary.scalar('training/energy', metrics['energy'] if metrics['energy'] is not None else 0, step=step)
    #                     tf.summary.scalar('memory/ram_used_mb', metrics['ram_used_mb'], step=step)
    #                     if 'gpu_memory_mb' in metrics:
    #                         tf.summary.scalar('memory/gpu_used_mb', metrics['gpu_memory_mb'], step=step)
    #                     tf.summary.text('notes/custom_message', metrics['notes'], step=step)
    #                 # [Integration] Log model weights and NaN counts after each outer iteration.
    #         if step==0:
    #             log_weights_and_nan_check(step, model_w, summary_writer)

    #         if step % 30 == 0:
    #             aggressive_memory_cleanup()
    #             tf.keras.backend.clear_session()
    #             gc.collect()                
    #             log_weights_and_nan_check(step, model_w, summary_writer)

    #             checkpoint_path = checkpoint_manager.save()
    #             print(f"Model checkpoint saved at step {step} to: {checkpoint_path}")
            
    #         # =============================
    #         # Updating the sampler_te.model
    #         # =============================
    #         copy_to_non_trainable(model_w, model_fix)

    #         # tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)
 
    #         for innerstep in range(hyperparams.sim_params.inner_loop):           
                
    #             inner_start_time = time.time()  # [Integration] Start timing inner iteration.
    #             for i in range(2*n_sites):
    #                 if hyperparams.simulation_type=="VMC2spins":
    #                     GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batchv2(model_w,GT_Batch_update)
    #                 elif hyperparams.simulation_type=="VMC":
    #                     GT_Batch_update, psi_new=sampler_var.monte_carlo_update_on_batch(model_w,GT_Batch_update)
    #             mc_duration=time.time()-inner_start_time
    #             phi_timer=time.time()
    #             phi_terms=compute_phi_terms(GT_Batch_update, sampler_te,model_fix)
    #             # print(f"phi_terms {phi_terms}, shape {phi_terms.shape}")
    #             phi_terms_duration=time.time()-phi_timer
    #             energy_time=time.time()
    #             energy, std_energy, loc_energies=stochastic_energy_tf(psi_new, model_w, edge_pairs, template_graphs_output, GT_Batch_update, 0.0)
    #             energy_duration=time.time()-energy_time
    #             # stoch_loss, stoch_grads=stochastic_overlap_gradient(phi_terms, GT_Batch_update, sampler_var)
    #             grad_start= time.time()
    #             stoch_loss, stoch_grads, stoch_overlap, grad_norms=improved_stochastic_gradients(phi_terms, GT_Batch_update, model_w)
                
    #             optimizer.apply(stoch_grads, model_w.trainable_variables)  
    #             grad_duration=time.time()-grad_start

    #             spins=GT_Batch_update.nodes[:,0]
    #             avg_spin=tf.reduce_mean(spins)    
    #             inner_end_time = time.time()  # [Integration] End timing inner iteration.
    #             with summary_writer.as_default():
    #                 tf.summary.scalar('timing/inner_step_duration', inner_end_time - inner_start_time,
    #                                 step=step * hyperparams.sim_params.inner_loop + innerstep)
    #                 tf.summary.scalar('timing/mc_duration', mc_duration,
    #                                 step=step * hyperparams.sim_params.inner_loop + innerstep)
    #                 tf.summary.scalar('timing/phi_terms_duration', phi_terms_duration,
    #                                 step=step * hyperparams.sim_params.inner_loop + innerstep)
    #                 tf.summary.scalar('timing/energy_duration', energy_duration,
    #                                 step=step * hyperparams.sim_params.inner_loop + innerstep)
    #                 tf.summary.scalar('timing/gradient_duration', grad_duration,    
    #                                 step=step * hyperparams.sim_params.inner_loop + innerstep)
                
    #             energies.append(tf.math.real.energy.numpy())
    #             std_energies.append(std_energy.numpy())
    #             magnetizations.append(avg_spin.numpy())
    #             # Clean up temporary variables to help with memory management.
    #             metrics = {
    #                 'energy': energies[-1] if energies else None,
    #                 'std_energy': std_energies[-1] if std_energies else None,
    #                 'magnetization': magnetizations[-1] if magnetizations else None,
    #                 'energy_imag': tf.math.imag(energy).numpy() if energy is not None else None,                    
    #                 # 'overlap': overlap_in_time[-1] if overlap_in_time else None,
    #                 'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
    #                 'notes': f"Outer step {step} completed"
    #             }  

    #             if innerstep==0:
    #                 log_training_metrics(summary_writer, step, metrics)    

    #             del stoch_loss, stoch_grads, energy, loc_energies, spins, avg_spin                
    #             del psi_new, phi_terms
    #     summary_writer.flush()
    #    
    #     outer_end_time = time.time()  # [Integration] End timing outer iteration.
    #     with summary_writer.as_default():
    #         tf.summary.scalar('timing/outer_step_duration', outer_end_time - outer_start_time, step=step)
    with summary_writer.as_default():
        tf.summary.text('configuration/hyperparameters', 
                    f'beta: {hyperparams.sim_params.beta}\nlearning_rate: {hyperparams.sim_params.learning_rate}\n', 
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
                tf.keras.backend.clear_session()
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
                        GT_Batch_update, psi_new = sampler_var.monte_carlo_update_on_batch(model_w, GT_Batch_update)
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
                avg_spin = tf.reduce_mean(spins)
                timing_metrics['inner_step_duration'] = time.time() - inner_start_time
                
                # Collect training metrics
                training_metrics = {
                    'energy_real': tf.math.real(energy),
                    'energy_imag': tf.math.imag(energy),
                    'std_energy': std_energy/tf.math.sqrt(tf.cast(loc_energies.shape[0], tf.float32)),
                    'magnetization': avg_spin,
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
                    if step % 5 == 0:
                        log_gradient_norms(global_step, stoch_grads, summary_writer)
                # Clean up temporary variables
                del stoch_loss, stoch_grads, energy, loc_energies, spins, avg_spin
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
