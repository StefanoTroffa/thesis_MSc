import os# Turn off XLA entirely
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
tf.keras.backend.clear_session()

# Also make sure the JIT optimizer is off
# tf.config.optimizer.set_jit(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled on {len(gpus)} devices")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

print("JIT enabled? ", tf.config.optimizer.get_jit(),'Was not it?')        # should be False

import numpy as np
import gc
import psutil
import matplotlib.pyplot as plt
import time
from datetime import datetime
from compgraph.monte_carlo import MCMCSampler, compute_phi_terms
from compgraph.tensorflow_version.hamiltonian_operations import stochastic_energy_tf, stochastic_gradients_tfv3
from compgraph.tensorflow_version.memory_control import aggressive_memory_cleanup
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure
from compgraph.useful import copy_to_non_trainable
import sonnet as snt
# Import the specific functions we want to profile
tf.debugging.set_log_device_placement(True)


def monitor_gpu():
    import subprocess
    result = subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True)
    return int(result.strip())

def track_all_operations(n_iterations=10, batch_size=8):
    """
    Track memory usage across all three key operations:
    1. compute_phi_terms
    2. stochastic_energy_tf
    3. stochastic_gradients_tfv3
    """
    print("Setting up environment for memory tracking...")

    from tensorflow.python.profiler import profiler_v2 as profiler

    # Set up options with emphasis on memory tracking
    profiler_options = profiler.ProfilerOptions(
        host_tracer_level=2,      # Most detailed host tracing
        python_tracer_level=1,    # Python function tracing
        device_tracer_level=1,    # Device (GPU) tracing
    )

 
    print("Setting up environment for memory tracking with profiling...")
    # Create minimal graph and model
    graph_type = "2dsquare"
    n,m = 6,6
    lattice_size = (n, m)
    sublattice = "Neel"
    beta = 0.005
    ansatz= "GNNprocnorm"
    # Create graph and model with minimal parameters
    graph, subl = create_graph_from_ham(graph_type, lattice_size, sublattice)
    model_params = {"hidden_size": 128, "output_emb_size": 64, "K_layer": 2}
    
    # Initialize two models
    model_w = initialize_NQS_model_fromhyperparams(ansatz, model_params, 860432)
    model_fix = initialize_NQS_model_fromhyperparams(ansatz, model_params)
    
    # Create graph tuples and edge pairs
    GT_Batch = initialize_graph_tuples_tf_opt(batch_size, graph, subl)
    senders, receivers, edge_pairs = precompute_graph_structure(graph)
    
    # Initialize models
    model_w(GT_Batch)
    model_fix(GT_Batch)
    
    # Copy weights from model_w to model_fix

    copy_to_non_trainable(model_w, model_fix)

    template_graphs_output=initialize_graph_tuples_tf_opt(tf.shape(edge_pairs)[0]+1,graph,subl)   
    # Create samplers
    sampler_var = MCMCSampler(GT_Batch, beta, edge_pairs=edge_pairs)
    sampler_te = MCMCSampler(GT_Batch, template=template_graphs_output, beta=beta,edge_pairs=edge_pairs)

    # Initialize optimizer for gradient application
    optimizer = snt.optimizers.Adam(1e-4)  # Use a small learning rate
    
    # Initialize tracking metrics
    process = psutil.Process()
    metrics = []
    
    # Force initial garbage collection
    aggressive_memory_cleanup()
    
    print("\nStarting comprehensive operation tracking...")
    print(f"Initial memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
        
    # Add TensorBoard writer initialization
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = f"logs/full_tracking/memory_tracking/{timestamp}"
    log_dir = f"logs/full_tracking/profiler/{ansatz}/{timestamp}"

    summary_writer = tf.summary.create_file_writer(log_dir)
    TRACE_EVERY = 100  # Capture full graph every 3 iterations
    
    print("\nStarting comprehensive operation tracking with graph tracing...")
    print(f"Initial memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"TensorBoard logs will be written to: {log_dir}")

    for i in range(n_iterations):
        print(f"\n=== Iteration {i+1}/{n_iterations} ===")
        # aggressive_memory_cleanup()
        # Only profile certain iterations to avoid excessive data

        # Pre-operations measurements
        pre_memory = process.memory_info().rss / (1024 * 1024)
                # Enable graph tracing periodically
        trace_this_iteration = (i % TRACE_EVERY == 0)
        trace_this_iteration=False

        should_profile = (i == 1) or (i == n_iterations - 1) or (trace_this_iteration)
        
        if should_profile:
            print(f"Starting profiler for iteration {i+1}...")
            profiler.start(log_dir, options=profiler_options)
        try:            
            # Step 0: Run monte_carlo_update_on_batch to prepare valid configurations
            print("Step 0: Running monte_carlo_update_on_batch...")
            start_time = time.time()

            for _ in range(n*m):
                GT_Batch, psi_new = sampler_var.monte_carlo_update_on_batch_profilemem(model_w, GT_Batch)
                        # Aggiungi questo per tracciare l'utilizzo reale durante l'esecuzione
            # esegui operazione
            mcmc_time = time.time() - start_time

            post_mcmc_memory = process.memory_info().rss / (1024 * 1024)
            # post_mcmc_tf_objects = count_tf_objects()
            
            print(f"MCMC update complete - Time: {mcmc_time:.3f}s, Memory: {pre_memory:.1f} → {post_mcmc_memory:.1f} MB")
            
            # Step 1: Compute phi_terms
            print("\nStep 1: Computing phi_terms...")
            start_time = time.time()
            mem_info_before_phi = tf.config.experimental.get_memory_info('GPU:0')

            phi_terms = compute_phi_terms(GT_Batch, sampler_te, model_fix)
            phi_terms_time = time.time() - start_time
            mem_info_after_phi = tf.config.experimental.get_memory_info('GPU:0')
            print(f"After phi_terms (nvidia-smi): {monitor_gpu()} MB")
            print(f"TF Memory After phi_terms: Current={mem_info_after_phi['current'] / (1024*1024):.2f} MiB, Peak={mem_info_after_phi['peak'] / (1024*1024):.2f} MiB")
            post_phi_memory = process.memory_info().rss / (1024 * 1024)

            with summary_writer.as_default():
                tf.summary.scalar('phi_terms/time', phi_terms_time, step=i)
                tf.summary.scalar('phi_terms/memory_diff', 
                                post_phi_memory - post_mcmc_memory, step=i)
                # tf.summary.scalar('phi_terms/tf_objects', 
                #                 post_phi_tf_objects - post_mcmc_tf_objects, step=i)    
            print(f"phi_terms complete - Time: {phi_terms_time:.3f}s, Memory: {post_mcmc_memory:.1f} → {post_phi_memory:.1f} MB")
            
            # Step 2: Compute stochastic energy
            print("\nStep 2: Computing stochastic_energy_tf...")
            tf.config.experimental.reset_memory_stats('GPU:0') # Reset peak counter
            print(f"Before Energy (nvidia-smi): {monitor_gpu()} MB")
            mem_info_before_energy = tf.config.experimental.get_memory_info('GPU:0')
            print(f"TF Memory Before Energy: Current={mem_info_before_energy['current'] / (1024*1024):.2f} MiB, Peak={mem_info_before_energy['peak'] / (1024*1024):.2f} MiB")
            start_time = time.time()
            energy, std, loc_energies = stochastic_energy_tf(psi_new, model_w, edge_pairs,template_graphs_output, GT_Batch, 0.0)
            energy_time = time.time() - start_time

            print(f"Dopo stoch energy: {monitor_gpu()} MB")
            mem_info_after_energy = tf.config.experimental.get_memory_info('GPU:0')
            print(f"TF Memory After Energy: Current={mem_info_after_energy['current'] / (1024*1024):.2f} MiB, Peak={mem_info_after_energy['peak'] / (1024*1024):.2f} MiB")
            print(f"After Energy (nvidia-smi): {monitor_gpu()} MB")
            print(f"TF Memory After Energy: Current={mem_info_after_energy['current'] / (1024*1024):.2f} MiB, Peak={mem_info_after_energy['peak'] / (1024*1024):.2f} MiB")

                    
            # Post-energy measurements
            post_energy_memory = process.memory_info().rss / (1024 * 1024)
            # post_energy_tf_objects = count_tf_objects()
            # Energy-specific logging
            with summary_writer.as_default():
                tf.summary.scalar('stochastic_energy/time', energy_time, step=i)
                tf.summary.scalar('stochastic_energy/value', energy.numpy(), step=i)
                tf.summary.scalar('stochastic_energy/memory_diff',
                                post_energy_memory - post_phi_memory, step=i)
            
            print(f"stochastic_energy complete - Time: {energy_time:.3f}s, Memory: {post_phi_memory:.1f} → {post_energy_memory:.1f} MB")
            print(f"Computed energy: {energy.numpy()}")
            
            # Step 3: Compute stochastic gradients

            # --- Right before the gradients step ---
            print("\nStep 3: Computing stochastic_gradients_tfv3...")
            tf.config.experimental.reset_memory_stats('GPU:0') # Reset peak counter
            print(f"Before Gradients (nvidia-smi): {monitor_gpu()} MB")
            mem_info_before = tf.config.experimental.get_memory_info('GPU:0')
            print(f"TF Memory Before Gradients: Current={mem_info_before['current'] / (1024*1024):.2f} MiB, Peak={mem_info_before['peak'] / (1024*1024):.2f} MiB")

            start_time = time.time()
            stoch_loss, stoch_grads = stochastic_gradients_tfv3(phi_terms, GT_Batch, model_w)
            gradients_time = time.time() - start_time
            
            # If it succeeds, capture memory info immediately
            mem_info_after = tf.config.experimental.get_memory_info('GPU:0')
            print(f"After Gradients (nvidia-smi): {monitor_gpu()} MB")
            print(f"TF Memory After Gradients: Current={mem_info_after['current'] / (1024*1024):.2f} MiB, Peak (during step)={mem_info_after['peak'] / (1024*1024):.2f} MiB")

            post_gradients_memory = process.memory_info().rss / (1024 * 1024)
            print(f"stochastic_gradients complete - Time: {gradients_time:.3f}s, Memory: {post_energy_memory:.1f} → {post_gradients_memory:.1f} MB")

            with summary_writer.as_default():
                tf.summary.scalar('stochastic_gradients/time', gradients_time, step=i)
                tf.summary.scalar('stochastic_gradients/loss', stoch_loss.numpy(), step=i)
                tf.summary.scalar('stochastic_gradients/memory_diff', post_gradients_memory - post_energy_memory, step=i)
                            
        
            

            # --- Inside the optimizer apply step ---
            print("\nApplying optimizer...")
            tf.config.experimental.reset_memory_stats('GPU:0') # Reset for optimizer step
            print(f"Before Optimizer Apply (nvidia-smi): {monitor_gpu()} MB")
            mem_info_before_opt = tf.config.experimental.get_memory_info('GPU:0')
            print(f"TF Memory Before Optimizer: Current={mem_info_before_opt['current'] / (1024*1024):.2f} MiB, Peak={mem_info_before_opt['peak'] / (1024*1024):.2f} MiB")

            start_time_opt = time.time()
            optimizer.apply(stoch_grads, model_w.trainable_variables)
            post_optimizer_time = time.time() - start_time_opt
            print(f"Optimizer apply complete - Time: {post_optimizer_time:.3f}s")
            mem_info_after_opt = tf.config.experimental.get_memory_info('GPU:0')
            print(f"After Optimizer Apply (nvidia-smi): {monitor_gpu()} MB")
            print(f"TF Memory After Optimizer: Current={mem_info_after_opt['current'] / (1024*1024):.2f} MiB, Peak (during step)={mem_info_after_opt['peak'] / (1024*1024):.2f} MiB")

            post_optimizer_memory = process.memory_info().rss / (1024 * 1024) # Host memory
            print(f"Optimizer applied - Memory: {post_gradients_memory:.1f} → {post_optimizer_memory:.1f} MB")


            with summary_writer.as_default():
                tf.summary.scalar('optimizer/memory_diff',
                                post_optimizer_memory - post_gradients_memory, step=i)

            del psi_new, phi_terms, energy, loc_energies, stoch_loss, stoch_grads


            post_deletion_memory = process.memory_info().rss / (1024 * 1024)
            print(f"Deletion complete - Memory: {post_optimizer_memory:.1f} → {post_deletion_memory:.1f} MB")

            with summary_writer.as_default():
                tf.summary.scalar('deletion/memory_diff', post_deletion_memory - post_optimizer_memory, step=i)
                    
            post_cleanup_memory = process.memory_info().rss / (1024 * 1024)


            print(f"Cleanup complete - Memory: {post_deletion_memory:.1f} → {post_cleanup_memory:.1f} MB")

            # Log cleanup impact
            with summary_writer.as_default():
                tf.summary.scalar('cleanup/memory_diff', post_cleanup_memory - post_deletion_memory, step=i)

            summary_writer.flush()

            metrics.append({
                'iteration': i+1,
                'pre_memory': pre_memory,
                'post_mcmc_memory': post_mcmc_memory,
                'mcmc_memory_diff': post_mcmc_memory - pre_memory,
                'mcmc_time': mcmc_time,
                'post_phi_memory': post_phi_memory,
                'phi_memory_diff': post_phi_memory - post_mcmc_memory,
                'phi_time': phi_terms_time,
                'post_energy_memory': post_energy_memory,
                'energy_memory_diff': post_energy_memory - post_phi_memory,
                'energy_time': energy_time,
                'post_gradients_memory': post_gradients_memory,
                'gradients_memory_diff': post_gradients_memory - post_energy_memory,
                'gradients_time': gradients_time,
                'post_optimizer_memory': post_optimizer_memory,
                'optimizer_memory_diff': post_optimizer_memory - post_gradients_memory,
                'optimizer_time': post_optimizer_time,
                'post_deletion_memory': post_deletion_memory,
                'deletion_memory_diff': post_deletion_memory - post_optimizer_memory,
                'post_cleanup_memory': post_cleanup_memory,
                'total_memory_increase': post_gradients_memory - pre_memory,
                'memory_retained_after_cleanup': post_cleanup_memory - pre_memory,

            })
            
            print(f"\n=== Iteration {i+1} Summary ===")
            print(f"Total memory impact: {metrics[-1]['total_memory_increase']:.2f} MB")
            print(f"Memory retained after cleanup: {metrics[-1]['memory_retained_after_cleanup']:.2f} MB")


        finally:
            # Stop profiler if active
            if should_profile:
                profiler.stop()
                print(f"Profiler data for iteration {i+1} saved to {log_dir}")

    plot_comprehensive_analysis(metrics)
    print_final_analysis(metrics)

    summary_writer.close()

    return metrics
def print_final_analysis(metrics):
    """
    Print final analysis of memory usage, separating the first iteration from the rest.
    Dynamically handles missing keys and excludes TensorFlow-related metrics if not present.
    """
    if not metrics:
        print("No metrics available for analysis.")
        return

    first_iteration = metrics[0]
    rest_iterations = metrics[1:] if len(metrics) > 1 else []

    print("\n=== Final Analysis ===")

    # Identify if TF-related metrics exist
    include_tf_metrics = 'post_cleanup_tf_objects' in first_iteration
    include_tf_functions = 'post_cleanup_tf_functions' in first_iteration

    def get_value(metric, key, default=0):
        return metric.get(key, default)

    # Overall Memory Metrics
    print("\n-- Overall Memory Metrics --")
    print(f"Average memory increase per iteration: {sum(get_value(m, 'total_memory_increase') for m in metrics)/len(metrics):.2f} MB")
    print(f"Final memory growth from start: {get_value(metrics[-1], 'post_cleanup_memory') - get_value(metrics[0], 'pre_memory'):.2f} MB")

    if include_tf_metrics:
        print(f"Final TF object growth from start: {get_value(metrics[-1], 'post_cleanup_tf_objects') - get_value(metrics[0], 'pre_tf_objects')}")

    # First Iteration Metrics
    print("\n-- First Iteration (With Graph Tracing) --")
    print(f"Memory increase: {get_value(first_iteration, 'total_memory_increase'):.2f} MB")
    print(f"Memory retained after cleanup: {get_value(first_iteration, 'memory_retained_after_cleanup'):.2f} MB")

    if include_tf_metrics:
        print(f"TF objects created: {get_value(first_iteration, 'post_gradients_tf_objects') - get_value(first_iteration, 'pre_tf_objects')}")
        print(f"TF objects retained after cleanup: {get_value(first_iteration, 'tf_objects_retained')}")

    # Memory Impact by Operation
    print("\nMemory impact by operation (First Iteration):")
    for key in ['mcmc_memory_diff', 'phi_memory_diff', 'energy_memory_diff', 'gradients_memory_diff', 'optimizer_memory_diff']:
        print(f"{key.replace('_', ' ').capitalize()}: {get_value(first_iteration, key):.2f} MB")

    if not rest_iterations:
        return

    # Steady-State Metrics (Subsequent Iterations)
    print("\n-- Subsequent Iterations (Steady State) --")
    avg_metrics = {
        key: sum(get_value(m, key, 0) for m in rest_iterations) / len(rest_iterations)
        for key in ['total_memory_increase', 'memory_retained_after_cleanup', 'mcmc_memory_diff', 'phi_memory_diff', 'energy_memory_diff', 'gradients_memory_diff', 'optimizer_memory_diff']
    }

    for key, value in avg_metrics.items():
        print(f"Average {key.replace('_', ' ')}: {value:.2f} MB")

    if include_tf_metrics:
        avg_tf_objects_created = sum(get_value(m, 'post_gradients_tf_objects') - get_value(m, 'pre_tf_objects') for m in rest_iterations) / len(rest_iterations)
        avg_tf_objects_retained = sum(get_value(m, 'tf_objects_retained') for m in rest_iterations) / len(rest_iterations)

        print(f"Average TF objects created: {avg_tf_objects_created:.2f}")
        print(f"Average TF objects retained after cleanup: {avg_tf_objects_retained:.2f}")

    # Memory impact comparison
    print("\n-- Comparison: First Iteration vs. Steady State --")
    for key in ['total_memory_increase', 'memory_retained_after_cleanup', 'mcmc_memory_diff', 'phi_memory_diff', 'energy_memory_diff', 'gradients_memory_diff', 'optimizer_memory_diff']:
        ratio = get_value(first_iteration, key, 1) / max(0.01, avg_metrics[key])
        print(f"{key.replace('_', ' ').capitalize()} ratio (First/Steady): {ratio:.2f}x")
    


def plot_comprehensive_analysis(metrics):
    """Create comprehensive visualizations of memory usage across all operations."""
    if not metrics:
        print("No metrics available for plotting.")
        return

    iterations = [m.get('iteration', i + 1) for i, m in enumerate(metrics)]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs('full_tracking', exist_ok=True)

    # Identify if TF-related metrics exist
    include_tf_metrics = 'post_cleanup_tf_objects' in metrics[0]
    include_tf_functions = 'post_cleanup_tf_functions' in metrics[0]

    def get_values(key):
        return [m.get(key, 0) for m in metrics]

    plt.figure(figsize=(18, 12))

    # Plot 1: Memory progression
    plt.subplot(2, 2, 1)
    plt.plot(iterations, get_values('pre_memory'), label='Initial')
    plt.plot(iterations, get_values('post_mcmc_memory'), label='After MCMC')
    plt.plot(iterations, get_values('post_phi_memory'), label='After phi_terms')
    plt.plot(iterations, get_values('post_energy_memory'), label='After energy')
    plt.plot(iterations, get_values('post_gradients_memory'), label='After gradients')
    plt.plot(iterations, get_values('post_optimizer_memory'), label='After optimizer')
    plt.plot(iterations, get_values('post_cleanup_memory'), label='After Cleanup')
    plt.title('Memory Progression (MB)')
    plt.xlabel('Iteration')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Memory impact by operation
    plt.subplot(2, 2, 2)
    width = 0.15
    x = np.array(iterations)
    plt.bar(x - 2*width, get_values('mcmc_memory_diff'), width, label='MCMC')
    plt.bar(x - width, get_values('phi_memory_diff'), width, label='phi_terms')
    plt.bar(x, get_values('energy_memory_diff'), width, label='Energy')
    plt.bar(x + width, get_values('gradients_memory_diff'), width, label='Gradients')
    plt.bar(x + 2*width, get_values('optimizer_memory_diff'), width, label='Optimizer')
    plt.title('Memory Impact by Operation (MB)')
    plt.xlabel('Iteration')
    plt.ylabel('Memory Change (MB)')
    plt.legend()
    plt.grid(True)

    # Plot 3: Memory retained after cleanup
    plt.subplot(2, 2, 3)
    plt.plot(iterations, get_values('memory_retained_after_cleanup'), 'r-', marker='o')
    plt.title('Memory Retained After Cleanup (MB)')
    plt.xlabel('Iteration')
    plt.ylabel('Memory (MB)')
    plt.grid(True)

    # Plot 4: TensorFlow objects retained (only if available)
    if include_tf_metrics:
        plt.subplot(2, 2, 4)
        plt.plot(iterations, get_values('tf_objects_retained'), 'b-', marker='o', label='TF Objects')
        if include_tf_functions:
            plt.plot(iterations, get_values('tf_functions_retained'), 'g-', marker='s', label='TF Functions')
        plt.title('TF Objects/Functions Retained')
        plt.xlabel('Iteration')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'full_tracking/memory_analysis_all_{timestamp}.png')
    print(f"Saved plot to 'full_tracking/memory_analysis_all_{timestamp}.png'")


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)

    print("TensorFlow Memory Analysis Tool")
    print("------------------------------")
    print("1. Comprehensive operation tracking")
    
    # Run comprehensive operation tracking
    n_iter = int(input("Number of iterations (default 10): ") or 10)
    batch_size = int(input("Batch size (default 32): ") or 32)
    track_all_operations(n_iterations=n_iter, batch_size=batch_size)
