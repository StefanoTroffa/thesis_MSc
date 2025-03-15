import tensorflow as tf
import numpy as np
import gc
import psutil
import matplotlib.pyplot as plt
import time
from datetime import datetime
from compgraph.monte_carlo import MCMCSampler, compute_phi_terms
from compgraph.tensorflow_version.hamiltonian_operations import stochastic_energy_tf, stochastic_gradients_tfv3
from compgraph.tensorflow_version.logging_tf import log_gradient_norms
from compgraph.tensorflow_version.memory_control import aggressive_memory_cleanup, count_tf_objects, inspect_tf_functions
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure
from compgraph.useful import copy_to_non_trainable
import sonnet as snt

# Import the specific functions we want to profile
from debug_malloc_tf import log_training_metrics, log_weights_and_nan_check
tf.debugging.set_log_device_placement(True)
# Set memory growth on GPUs if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled on {len(gpus)} devices")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

# Function to track graph/tensor objects in TensorFlow
# Function to inspect internal TensorFlow graphs
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
        host_tracer_level=3,      # Most detailed host tracing
        python_tracer_level=1,    # Python function tracing
        device_tracer_level=1,    # Device (GPU) tracing
    )

 
    print("Setting up environment for memory tracking with profiling...")
    # Create minimal graph and model
    graph_type = "2dsquare"
    lattice_size = (2, 2)
    sublattice = "Neel"
    beta = 0.01
    
    # Create graph and model with minimal parameters
    graph, subl = create_graph_from_ham(graph_type, lattice_size, sublattice)
    model_params = {"hidden_size": 32, "output_emb_size": 8}
    
    # Initialize two models
    model_w = initialize_NQS_model_fromhyperparams("GNN2simple", model_params)
    model_fix = initialize_NQS_model_fromhyperparams("GNN2simple", model_params)
    
    # Create graph tuples and edge pairs
    GT_Batch = initialize_graph_tuples_tf_opt(batch_size, graph, subl)
    senders, receivers, edge_pairs = precompute_graph_structure(graph)
    
    # Initialize models
    model_w(GT_Batch)
    model_fix(GT_Batch)
    
    # Copy weights from model_w to model_fix
    copy_to_non_trainable(model_w, model_fix)
    
    # Create samplers
    sampler_var = MCMCSampler(model_w, GT_Batch, beta, edge_pairs=edge_pairs)
    sampler_te = MCMCSampler(model_fix, GT_Batch, beta, edge_pairs=edge_pairs)
    
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
    log_dir = f"logs/full_tracking/profiler/{timestamp}"

    summary_writer = tf.summary.create_file_writer(log_dir)
    TRACE_EVERY = 10  # Capture full graph every 3 iterations
    
    print("\nStarting comprehensive operation tracking with graph tracing...")
    print(f"Initial memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"TensorBoard logs will be written to: {log_dir}")

    for i in range(n_iterations):
        print(f"\n=== Iteration {i+1}/{n_iterations} ===")
        aggressive_memory_cleanup()
        # Only profile certain iterations to avoid excessive data

        # Pre-operations measurements
        pre_memory = process.memory_info().rss / (1024 * 1024)
        pre_tf_objects = count_tf_objects()
        pre_tf_functions = inspect_tf_functions()
                # Enable graph tracing periodically
        trace_this_iteration = (i % TRACE_EVERY == 0)
        trace_this_iteration=False
        # trace_this_iteration_complete = (i % TRACE_EVERY == TRACE_EVERY - 1)
        # if trace_this_iteration_complete:
        #     print(f"Enabling full graph tracing for iteration {i+1}")
        #     tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)

        should_profile = (i == 1) or (i == n_iterations - 1) or (trace_this_iteration)
        
        if should_profile:
            print(f"Starting profiler for iteration {i+1}...")
            profiler.start(log_dir, options=profiler_options)
        try:            
            # Step 0: Run monte_carlo_update_on_batch to prepare valid configurations
            print("Step 0: Running monte_carlo_update_on_batch...")
            start_time = time.time()
            # Conditionally trace the MCMC component
            if trace_this_iteration:
                with summary_writer.as_default():
                    tf.summary.trace_on(graph=True, profiler=False)
                
            GT_Batch, psi_new = sampler_var.monte_carlo_update_on_batch(GT_Batch, 4)
            
            # Export MCMC component trace
            if trace_this_iteration:
                with summary_writer.as_default():
                    tf.summary.trace_export(
                        name=f"mcmc_graph",
                        step=i)
            
            mcmc_time = time.time() - start_time
            # with tf.name_scope("monte_carlo_update"):  # Add name scope for better visualization
            #         start_time = time.time()
            #         GT_Batch, psi_new = sampler_var.monte_carlo_update_on_batch(GT_Batch, 20)
            #         mcmc_time = time.time() - start_time
            # Post-MCMC measurements
            post_mcmc_memory = process.memory_info().rss / (1024 * 1024)
            post_mcmc_tf_objects = count_tf_objects()
            
            print(f"MCMC update complete - Time: {mcmc_time:.3f}s, Memory: {pre_memory:.1f} → {post_mcmc_memory:.1f} MB")
            
            # Step 1: Compute phi_terms
            print("\nStep 1: Computing phi_terms...")
            start_time = time.time()
            if trace_this_iteration:
                with summary_writer.as_default():
                    tf.summary.trace_on(graph=True, profiler=False)
            
            phi_terms = compute_phi_terms(GT_Batch, sampler_te)
            
            # # Export phi_terms component trace
            if trace_this_iteration:
                with summary_writer.as_default():
                    tf.summary.trace_export(
                        name=f"phi_terms_graph",
                        step=i)
            phi_terms_time = time.time() - start_time
            # with tf.name_scope("phi_terms_compute"):
            #     start_time = time.time()
            #     phi_terms = compute_phi_terms(GT_Batch, sampler_te)
            #     phi_terms_time = time.time() - start_time
            
            # Log phi_terms metrics
            post_phi_memory = process.memory_info().rss / (1024 * 1024)
            post_phi_tf_objects = count_tf_objects()

            with summary_writer.as_default():
                tf.summary.scalar('phi_terms/time', phi_terms_time, step=i)
                tf.summary.scalar('phi_terms/memory_diff', 
                                post_phi_memory - post_mcmc_memory, step=i)
                tf.summary.scalar('phi_terms/tf_objects', 
                                post_phi_tf_objects - post_mcmc_tf_objects, step=i)    
            print(f"phi_terms complete - Time: {phi_terms_time:.3f}s, Memory: {post_mcmc_memory:.1f} → {post_phi_memory:.1f} MB")
            
            # Step 2: Compute stochastic energy
            print("\nStep 2: Computing stochastic_energy_tf...")
            start_time = time.time()
            # # Conditionally trace the energy component
            if trace_this_iteration:
                with summary_writer.as_default():
                    tf.summary.trace_on(graph=True, profiler=False)
            
            energy, loc_energies = stochastic_energy_tf(psi_new, sampler_var, edge_pairs, GT_Batch, 0.0)
            
            # # Export energy component trace
            if trace_this_iteration:
                with summary_writer.as_default():
                    tf.summary.trace_export(
                        name=f"energy_graph",
                        step=i)
            energy_time = time.time() - start_time
            # print("\nStep 2: Computing stochastic_energy_tf...")
            # with tf.name_scope("energy_compute"):
            #     start_time = time.time()
            #     energy, loc_energies = stochastic_energy_tf(psi_new, sampler_var, edge_pairs, GT_Batch, 0.0)
            #     energy_time = time.time() - start_time
                    
            # Post-energy measurements
            post_energy_memory = process.memory_info().rss / (1024 * 1024)
            post_energy_tf_objects = count_tf_objects()
            # Energy-specific logging
            with summary_writer.as_default():
                tf.summary.scalar('stochastic_energy/time', energy_time, step=i)
                tf.summary.scalar('stochastic_energy/value', energy.numpy(), step=i)
                tf.summary.scalar('stochastic_energy/memory_diff',
                                post_energy_memory - post_phi_memory, step=i)
            
            print(f"stochastic_energy complete - Time: {energy_time:.3f}s, Memory: {post_phi_memory:.1f} → {post_energy_memory:.1f} MB")
            print(f"Computed energy: {energy.numpy()}")
            
            # Step 3: Compute stochastic gradients
            # Step 3: Compute stochastic gradients
            print("\nStep 3: Computing stochastic_gradients_tfv3...")
            # with tf.name_scope("gradient_compute"):
            #     start_time = time.time()
            #     stoch_loss, stoch_grads = stochastic_gradients_tfv3(phi_terms, GT_Batch, sampler_var)
            #     gradients_time = time.time() - start_time
            start_time = time.time()
            # # Conditionally trace the gradients component
            if trace_this_iteration:
                with summary_writer.as_default():
                    tf.summary.trace_on(graph=True, profiler=False)
            
            stoch_loss, stoch_grads = stochastic_gradients_tfv3(phi_terms, GT_Batch, sampler_var)
            
            # # Export gradients component trace
            if trace_this_iteration:
                with summary_writer.as_default():
                    tf.summary.trace_export(
                        name=f"gradients_graph",
                        step=i)
            gradients_time = time.time() - start_time
            
            # Post-gradients measurements
            post_gradients_memory = process.memory_info().rss / (1024 * 1024)
            post_gradients_tf_objects = count_tf_objects()
                # Gradient-specific logging
            with summary_writer.as_default():
                tf.summary.scalar('stochastic_gradients/time', gradients_time, step=i)
                tf.summary.scalar('stochastic_gradients/loss', stoch_loss.numpy(), step=i)
                tf.summary.scalar('stochastic_gradients/memory_diff', post_gradients_memory - post_energy_memory, step=i)
                tf.summary.scalar('stochastic_gradients/tf_objects', post_gradients_tf_objects - post_energy_tf_objects, step=i)
                            
                # log_gradient_norms(i, stoch_grads, summary_writer)
        
            print(f"stochastic_gradients complete - Time: {gradients_time:.3f}s, Memory: {post_energy_memory:.1f} → {post_gradients_memory:.1f} MB")
            
            # Apply gradients (optional, to simulate real training)
            # # Apply gradients
            # with tf.name_scope("optimizer_apply"):
            #     optimizer.apply(stoch_grads, sampler_var.model.trainable_variables) 
            start_time_opt = time.time()
            optimizer.apply(stoch_grads, sampler_var.model.trainable_variables)            
            post_optimizer_time = time.time() - start_time_opt  
            post_optimizer_memory = process.memory_info().rss / (1024 * 1024)
            post_optimizer_tf_objects = count_tf_objects()
            print(f"Optimizer applied - Memory: {post_gradients_memory:.1f} → {post_optimizer_memory:.1f} MB")

            # print(locals())
            # Export full graph trace if enabled
            # if trace_this_iteration_complete:
            #     with summary_writer.as_default():
            #         tf.summary.trace_export(
            #             name=f"full_graph_iter_{i}",
            #             step=i)        


            # Optimizer-specific logging
            with summary_writer.as_default():
                tf.summary.scalar('optimizer/memory_diff',
                                post_optimizer_memory - post_gradients_memory, step=i)
                # log_weights_and_nan_check(i, sampler_var.model, summary_writer)
            # Clear references
            del psi_new, phi_terms, energy, loc_energies, stoch_loss, stoch_grads


            post_deletion_memory = process.memory_info().rss / (1024 * 1024)
            post_deletion_tf_objects = count_tf_objects()
            print(f"Deletion complete - Memory: {post_optimizer_memory:.1f} → {post_deletion_memory:.1f} MB")
            # After cleanup measurements
            # aggressive_memory_cleanup()
            # Log deletion impact
            with summary_writer.as_default():
                tf.summary.scalar('deletion/memory_diff', post_deletion_memory - post_optimizer_memory, step=i)
                tf.summary.scalar('deletion/tf_objects', post_deletion_tf_objects - post_optimizer_tf_objects, step=i)
                    
            post_cleanup_memory = process.memory_info().rss / (1024 * 1024)
            post_cleanup_tf_objects = count_tf_objects()


            print(f"Cleanup complete - Memory: {post_deletion_memory:.1f} → {post_cleanup_memory:.1f} MB")

            post_cleanup_tf_functions = inspect_tf_functions()
            
            # Log cleanup impact
            with summary_writer.as_default():
                tf.summary.scalar('cleanup/memory_diff', post_cleanup_memory - post_deletion_memory, step=i)
                tf.summary.scalar('cleanup/tf_objects', post_cleanup_tf_objects - post_deletion_tf_objects, step=i)
                tf.summary.scalar('cleanup/tf_functions', post_cleanup_tf_functions, step=i)
            summary_writer.flush()

                # Log graph-memory correlations
                # if trace_this_iteration:
                #     tf.summary.scalar('graph_stats/memory_per_operation',
                #                      (post_cleanup_memory - pre_memory) / max(1, post_cleanup_tf_objects - pre_tf_objects),
                #                      step=i)
                            
            # Record complete metrics for this iteration
            metrics.append({
                'iteration': i+1,
                # Pre-operation memory state
                'pre_memory': pre_memory,
                'pre_tf_objects': pre_tf_objects,
                'pre_tf_functions': pre_tf_functions,
                
                # Post-MCMC memory state
                'post_mcmc_memory': post_mcmc_memory,
                'post_mcmc_tf_objects': post_mcmc_tf_objects,
                'mcmc_memory_diff': post_mcmc_memory - pre_memory,
                'mcmc_time': mcmc_time,
                
                # Post-phi_terms memory state
                'post_phi_memory': post_phi_memory,
                'post_phi_tf_objects': post_phi_tf_objects,
                'phi_memory_diff': post_phi_memory - post_mcmc_memory,
                'phi_time': phi_terms_time,
                
                # Post-energy memory state
                'post_energy_memory': post_energy_memory,
                'post_energy_tf_objects': post_energy_tf_objects,
                'energy_memory_diff': post_energy_memory - post_phi_memory,
                'energy_time': energy_time,
                
                # Post-gradients memory state
                'post_gradients_memory': post_gradients_memory,
                'post_gradients_tf_objects': post_gradients_tf_objects,
                'gradients_memory_diff': post_gradients_memory - post_energy_memory,
                'gradients_time': gradients_time,
                

                #Post memory state after optimizer
                'post_optimizer_memory': post_optimizer_memory,
                'post_optimizer_tf_objects': post_optimizer_tf_objects,
                'optimizer_memory_diff': post_optimizer_memory - post_gradients_memory,
                'optimizer_time': post_optimizer_time,
                #Post memory state after deletion   
                'post_deletion_memory': post_deletion_memory,
                'post_deletion_tf_objects': post_deletion_tf_objects,
                'deletion_memory_diff': post_deletion_memory - post_optimizer_memory,


                # Post-cleanup memory state
                'post_cleanup_memory': post_cleanup_memory,
                'post_cleanup_tf_objects': post_cleanup_tf_objects,
                'post_cleanup_tf_functions': post_cleanup_tf_functions,
                

                # Total impact
                'total_memory_increase': post_gradients_memory - pre_memory,
                'memory_retained_after_cleanup': post_cleanup_memory - pre_memory,
                'tf_objects_retained': post_cleanup_tf_objects - pre_tf_objects,
                'tf_functions_retained': post_cleanup_tf_functions - pre_tf_functions
            })
            
            # Print summary for this iteration
            print(f"\n=== Iteration {i+1} Summary ===")
            print(f"Total memory impact: {metrics[-1]['total_memory_increase']:.2f} MB")
            print(f"Memory retained after cleanup: {metrics[-1]['memory_retained_after_cleanup']:.2f} MB")
            print(f"TF objects retained: {metrics[-1]['tf_objects_retained']}")
            print(f"TF functions retained: {metrics[-1]['tf_functions_retained']}")

        finally:
            # Stop profiler if active
            if should_profile:
                profiler.stop()
                print(f"Profiler data for iteration {i+1} saved to {log_dir}")

    # Create visualizations
    plot_comprehensive_analysis(metrics)
    # With this:
    print_final_analysis(metrics)
    # Print final summary
    # print("\n=== Final Analysis ===")
    # print(f"Average memory increase per iteration: {sum(m['total_memory_increase'] for m in metrics)/len(metrics):.2f} MB")
    # print(f"Average memory retained after cleanup: {sum(m['memory_retained_after_cleanup'] for m in metrics)/len(metrics):.2f} MB")
    # print(f"Final memory growth from start: {metrics[-1]['post_cleanup_memory'] - metrics[0]['pre_memory']:.2f} MB")
    # print(f"Final TF object growth from start: {metrics[-1]['post_cleanup_tf_objects'] - metrics[0]['pre_tf_objects']}")
    
    # # Identify the operation with highest memory impact
    # avg_mcmc_impact = sum(m['mcmc_memory_diff'] for m in metrics)/len(metrics)
    # avg_phi_impact = sum(m['phi_memory_diff'] for m in metrics)/len(metrics)
    # avg_energy_impact = sum(m['energy_memory_diff'] for m in metrics)/len(metrics)
    # avg_gradients_impact = sum(m['gradients_memory_diff'] for m in metrics)/len(metrics)
    
    # print("\nMemory impact by operation:")
    # print(f"MCMC update: {avg_mcmc_impact:.2f} MB")
    # print(f"phi_terms: {avg_phi_impact:.2f} MB")
    # print(f"stochastic_energy: {avg_energy_impact:.2f} MB")
    # print(f"stochastic_gradients: {avg_gradients_impact:.2f} MB")
    summary_writer.close()

    return metrics
def print_final_analysis(metrics):
    """
    Print final analysis of memory usage, separating first iteration from the rest.
    This helps differentiate between initial graph tracing overhead and steady-state behavior.
    """
    # Separate first iteration from the rest
    first_iteration = metrics[0]
    rest_iterations = metrics[1:] if len(metrics) > 1 else []
    
    print("\n=== Final Analysis ===")
    
    # Overall metrics
    print("\n-- Overall Metrics --")
    print(f"Average memory increase per iteration: {sum(m['total_memory_increase'] for m in metrics)/len(metrics):.2f} MB")
    print(f"Average memory retained after cleanup: {sum(m['memory_retained_after_cleanup'] for m in metrics)/len(metrics):.2f} MB")
    print(f"Final memory growth from start: {metrics[-1]['post_cleanup_memory'] - metrics[0]['pre_memory']:.2f} MB")
    print(f"Final TF object growth from start: {metrics[-1]['post_cleanup_tf_objects'] - metrics[0]['pre_tf_objects']}")
    
    # First iteration metrics (with graph tracing)
    print("\n-- First Iteration (With Graph Tracing) --")
    print(f"Memory increase: {first_iteration['total_memory_increase']:.2f} MB")
    print(f"Memory retained after cleanup: {first_iteration['memory_retained_after_cleanup']:.2f} MB")
    print(f"TF objects created: {first_iteration['post_gradients_tf_objects'] - first_iteration['pre_tf_objects']}")
    print(f"TF objects retained after cleanup: {first_iteration['tf_objects_retained']}")
    
    # Memory impact by operation for first iteration
    print("\nMemory impact by operation (First Iteration):")
    print(f"MCMC update: {first_iteration['mcmc_memory_diff']:.2f} MB")
    print(f"phi_terms: {first_iteration['phi_memory_diff']:.2f} MB")
    print(f"stochastic_energy: {first_iteration['energy_memory_diff']:.2f} MB")
    print(f"stochastic_gradients: {first_iteration['gradients_memory_diff']:.2f} MB")
    print(f"optimizer: {first_iteration['optimizer_memory_diff']:.2f} MB")
    
    # Skip further analysis if there's only one iteration
    if not rest_iterations:
        return
    
    # Subsequent iterations metrics (steady state)
    print("\n-- Subsequent Iterations (Steady State) --")
    avg_memory_increase = sum(m['total_memory_increase'] for m in rest_iterations)/len(rest_iterations)
    avg_memory_retained = sum(m['memory_retained_after_cleanup'] for m in rest_iterations)/len(rest_iterations)
    avg_tf_objects_created = sum(m['post_gradients_tf_objects'] - m['pre_tf_objects'] for m in rest_iterations)/len(rest_iterations)
    avg_tf_objects_retained = sum(m['tf_objects_retained'] for m in rest_iterations)/len(rest_iterations)
    
    print(f"Average memory increase: {avg_memory_increase:.2f} MB")
    print(f"Average memory retained after cleanup: {avg_memory_retained:.2f} MB")
    print(f"Average TF objects created: {avg_tf_objects_created:.2f}")
    print(f"Average TF objects retained after cleanup: {avg_tf_objects_retained:.2f}")
    
    # Memory impact by operation for subsequent iterations
    avg_mcmc_impact = sum(m['mcmc_memory_diff'] for m in rest_iterations)/len(rest_iterations)
    avg_phi_impact = sum(m['phi_memory_diff'] for m in rest_iterations)/len(rest_iterations)
    avg_energy_impact = sum(m['energy_memory_diff'] for m in rest_iterations)/len(rest_iterations)
    avg_gradients_impact = sum(m['gradients_memory_diff'] for m in rest_iterations)/len(rest_iterations)
    avg_optimizer_impact = sum(m['optimizer_memory_diff'] for m in rest_iterations)/len(rest_iterations)
    
    print("\nMemory impact by operation (Steady State):")
    print(f"MCMC update: {avg_mcmc_impact:.2f} MB")
    print(f"phi_terms: {avg_phi_impact:.2f} MB")
    print(f"stochastic_energy: {avg_energy_impact:.2f} MB")
    print(f"stochastic_gradients: {avg_gradients_impact:.2f} MB")
    print(f"optimizer: {avg_optimizer_impact:.2f} MB")
    
    # Comparison between first iteration and steady state
    print("\n-- Comparison: First Iteration vs. Steady State --")
    print("Memory increase ratio (First/Steady): " + 
          f"{first_iteration['total_memory_increase']/max(0.01, avg_memory_increase):.2f}x")
    print("Memory retained ratio (First/Steady): " + 
          f"{first_iteration['memory_retained_after_cleanup']/max(0.01, avg_memory_retained):.2f}x")
    print("TF objects created ratio (First/Steady): " +
          f"{(first_iteration['post_gradients_tf_objects'] - first_iteration['pre_tf_objects'])/max(1, avg_tf_objects_created):.2f}x")
    
    # Operation-specific ratios
    print("\nOperation memory impact ratios (First/Steady):")
    print(f"MCMC update: {first_iteration['mcmc_memory_diff']/max(0.01, avg_mcmc_impact):.2f}x")
    print(f"phi_terms: {first_iteration['phi_memory_diff']/max(0.01, avg_phi_impact):.2f}x")
    print(f"stochastic_energy: {first_iteration['energy_memory_diff']/max(0.01, avg_energy_impact):.2f}x")
    print(f"stochastic_gradients: {first_iteration['gradients_memory_diff']/max(0.01, avg_gradients_impact):.2f}x")
    print(f"optimizer: {first_iteration['optimizer_memory_diff']/max(0.01, avg_optimizer_impact):.2f}x")
    
def plot_comprehensive_analysis(metrics):
    """Create comprehensive visualizations of memory usage across all operations including optimizer impact"""
    iterations = [m['iteration'] for m in metrics]
    
    # Create a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create a larger figure to accommodate all plots
    plt.figure(figsize=(24, 20))
    
    # === Original six plots ===
    
    # Plot 1: Memory progression across all operations
    plt.subplot(4, 2, 1)
    plt.plot(iterations, [m['pre_memory'] for m in metrics], 'k-', label='Initial')
    plt.plot(iterations, [m['post_mcmc_memory'] for m in metrics], 'b-', label='After MCMC')
    plt.plot(iterations, [m['post_phi_memory'] for m in metrics], 'g-', label='After phi_terms')
    plt.plot(iterations, [m['post_energy_memory'] for m in metrics], 'y-', label='After energy')
    plt.plot(iterations, [m['post_gradients_memory'] for m in metrics], 'r-', label='After gradients')
    plt.plot(iterations, [m['post_optimizer_memory'] for m in metrics], 'm-', label='After optimizer')
    plt.plot(iterations, [m['post_cleanup_memory'] for m in metrics], 'k--', label='After Cleanup')
    plt.title('Memory Progression Through Operations (MB)')
    plt.xlabel('Iteration')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Memory impact of each operation
    plt.subplot(4, 2, 2)
    width = 0.15
    x = np.array(iterations)
    plt.bar(x - 2*width, [m['mcmc_memory_diff'] for m in metrics], width, label='MCMC')
    plt.bar(x - width, [m['phi_memory_diff'] for m in metrics], width, label='phi_terms')
    plt.bar(x, [m['energy_memory_diff'] for m in metrics], width, label='Energy')
    plt.bar(x + width, [m['gradients_memory_diff'] for m in metrics], width, label='Gradients')
    plt.bar(x + 2*width, [m['optimizer_memory_diff'] for m in metrics], width, label='Optimizer')
    plt.title('Memory Impact by Operation (MB)')
    plt.xlabel('Iteration')
    plt.ylabel('Memory Change (MB)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: TF object count progression
    plt.subplot(4, 2, 3)
    plt.plot(iterations, [m['pre_tf_objects'] for m in metrics], 'k-', label='Initial')
    plt.plot(iterations, [m['post_mcmc_tf_objects'] for m in metrics], 'b-', label='After MCMC')
    plt.plot(iterations, [m['post_phi_tf_objects'] for m in metrics], 'g-', label='After phi_terms')
    plt.plot(iterations, [m['post_energy_tf_objects'] for m in metrics], 'y-', label='After energy')
    plt.plot(iterations, [m['post_gradients_tf_objects'] for m in metrics], 'r-', label='After gradients')
    plt.plot(iterations, [m['post_optimizer_tf_objects'] for m in metrics], 'm-', label='After optimizer')
    plt.plot(iterations, [m['post_cleanup_tf_objects'] for m in metrics], 'k--', label='After Cleanup')
    plt.title('TensorFlow Object Count Progression')
    plt.xlabel('Iteration')
    plt.ylabel('TF Object Count')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Execution time comparison
    plt.subplot(4, 2, 4)
    width = 0.15
    plt.bar(x - 2*width, [m['mcmc_time'] for m in metrics], width, label='MCMC')
    plt.bar(x - width, [m['phi_time'] for m in metrics], width, label='phi_terms')
    plt.bar(x, [m['energy_time'] for m in metrics], width, label='Energy')
    plt.bar(x + width, [m['gradients_time'] for m in metrics], width, label='Gradients')
    plt.title('Execution Time by Operation (seconds)')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Memory retained after cleanup
    plt.subplot(4, 2, 5)
    plt.plot(iterations, [m['memory_retained_after_cleanup'] for m in metrics], 'r-', marker='o')
    plt.title('Memory Retained After Cleanup (MB)')
    plt.xlabel('Iteration')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    
    # Plot 6: TF objects retained after cleanup
    plt.subplot(4, 2, 6)
    plt.plot(iterations, [m['tf_objects_retained'] for m in metrics], 'b-', marker='o', label='Objects')
    plt.plot(iterations, [m['tf_functions_retained'] for m in metrics], 'g-', marker='s', label='Functions')
    plt.title('TF Objects/Functions Retained')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # === New plots for optimizer analysis ===
    
    # Plot 7: Memory change from optimizer application and deletion
    plt.subplot(4, 2, 7)
    width = 0.3
    plt.bar(x - width/2, [m['optimizer_memory_diff'] for m in metrics], width, label='Optimizer Impact')
    plt.bar(x + width/2, [m['deletion_memory_diff'] for m in metrics], width, label='Deletion Impact')
    plt.title('Memory Impact: Optimizer Application vs Variable Deletion (MB)')
    plt.xlabel('Iteration')
    plt.ylabel('Memory Change (MB)')
    plt.legend()
    plt.grid(True)
    
    # Plot 8: Optimizer percentage contribution
    plt.subplot(4, 2, 8)
    total_memory_impacts = [
        m['mcmc_memory_diff'] + m['phi_memory_diff'] + 
        m['energy_memory_diff'] + m['gradients_memory_diff'] + 
        m['optimizer_memory_diff']
        for m in metrics
    ]
    optimizer_percentages = [
        100 * m['optimizer_memory_diff'] / max(0.1, total) 
        for m, total in zip(metrics, total_memory_impacts)
    ]
    plt.bar(iterations, optimizer_percentages, color='magenta')
    plt.title('Optimizer Contribution to Total Memory Impact (%)')
    plt.xlabel('Iteration')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the comprehensive figure
    import os
    os.makedirs('full_tracking', exist_ok=True)
    plt.savefig(f'full_tracking/memory_analysis_all_{timestamp}.png')
    
    # Create a separate figure for the stacked bar chart
    plt.figure(figsize=(15, 10))
    
    # Prepare data for stacked bar chart
    labels = ['MCMC', 'phi_terms', 'Energy', 'Gradients', 'Optimizer']
    x_labels = [f'Iter {i}' for i in iterations]
    
    mcmc_diffs = [m['mcmc_memory_diff'] for m in metrics]
    phi_diffs = [m['phi_memory_diff'] for m in metrics]
    energy_diffs = [m['energy_memory_diff'] for m in metrics]
    gradient_diffs = [m['gradients_memory_diff'] for m in metrics]
    optimizer_diffs = [m['optimizer_memory_diff'] for m in metrics]
    
    # Create stacked bar chart
    width = 0.7
    bottom_vals = np.zeros(len(iterations))
    
    # For each operation, add its memory diff on top of previous operations
    for diffs, label, color in zip(
        [mcmc_diffs, phi_diffs, energy_diffs, gradient_diffs, optimizer_diffs],
        labels,
        ['blue', 'green', 'yellow', 'red', 'magenta']
    ):
        plt.bar(
            x_labels, 
            diffs, 
            width, 
            bottom=bottom_vals, 
            label=label,
            color=color
        )
        bottom_vals = bottom_vals + np.array(diffs)
    
    plt.title('Cumulative Memory Impact by Operation Stage')
    plt.xlabel('Iteration')
    plt.ylabel('Memory Change (MB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the stacked bar chart
    plt.savefig(f'full_tracking/memory_cumulative_{timestamp}.png')
    
    print(f"Saved all plots to 'full_tracking/memory_analysis_all_{timestamp}.png'")
    print(f"Saved cumulative memory chart to 'full_tracking/memory_cumulative_{timestamp}.png'")

def run_accumulation_test(n_iterations=3, inner_iterations=5, batch_size=32):
    """
    Run multiple rounds of all operations to simulate training loop behavior.
    Each round consists of multiple iterations of the operations without session clearing.
    After each round, we perform aggressive cleanup.
    
    This helps identify memory accumulation patterns that occur in actual training.
    """
    print("Setting up environment for memory accumulation testing...")
    
    # Create minimal graph and model
    graph_type = "2dsquare"
    lattice_size = (3, 3)
    sublattice = "Neel"
    beta = 0.06
    
    # Create graph and model with minimal parameters
    graph, subl = create_graph_from_ham(graph_type, lattice_size, sublattice)
    model_params = {"hidden_size": 128, "output_emb_size": 64}
    
    # Initialize models
    model_w = initialize_NQS_model_fromhyperparams("GNN2simple", model_params)
    model_fix = initialize_NQS_model_fromhyperparams("GNN2simple", model_params)
    
    # Create graph tuples and edge pairs
    GT_Batch = initialize_graph_tuples_tf_opt(batch_size, graph, subl)
    senders, receivers, edge_pairs = precompute_graph_structure(graph)
    
    # Initialize models
    model_w(GT_Batch)
    model_fix(GT_Batch)
    
    # Copy weights from model_w to model_fix
    copy_to_non_trainable(model_w, model_fix)
    
    # Create samplers
    sampler_var = MCMCSampler(model_w, GT_Batch, beta, edge_pairs=edge_pairs)
    sampler_te = MCMCSampler(model_fix, GT_Batch, beta, edge_pairs=edge_pairs)
    
    # Initialize optimizer
    optimizer = snt.optimizers.Adam(1e-4)
    
    # Initialize tracking metrics
    process = psutil.Process()
    metrics = []
    
    # Force initial garbage collection
    aggressive_memory_cleanup()
    
    print("\nStarting memory accumulation testing...")
    print(f"Initial memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    
    for round_idx in range(n_iterations):
        print(f"\n=== Round {round_idx+1}/{n_iterations} ===")
        
        # Pre-round measurements
        pre_round_memory = process.memory_info().rss / (1024 * 1024)
        pre_round_tf_objects = count_tf_objects()
        pre_round_tf_functions = inspect_tf_functions()
        
        round_metrics = {
            'round': round_idx+1,
            'pre_memory': pre_round_memory,
            'pre_tf_objects': pre_round_tf_objects,
            'pre_tf_functions': pre_round_tf_functions,
            'iterations': []
        }
        gc.collect()
        copy_to_non_trainable(sampler_var.model, sampler_te.model)

        # Run inner iterations without aggressive cleanup between them
        for inner_idx in range(inner_iterations):
            print(f"\n-- Inner iteration {inner_idx+1}/{inner_iterations} --")
            
            # Perform minimal cleanup to remove previous iteration's variables
            
            # Pre-iteration measurements
            pre_iter_memory = process.memory_info().rss / (1024 * 1024)
            pre_iter_tf_objects = count_tf_objects()
            
            # Step 1: Run monte_carlo_update_on_batch
            print("Step 1: Running monte_carlo_update_on_batch...")
            start_time = time.time()
            GT_Batch, psi_new = sampler_var.monte_carlo_update_on_batch(GT_Batch, 20)
            mcmc_time = time.time() - start_time
            
            # Step 2: Compute phi_terms
            print("Step 2: Computing phi_terms...")
            start_time = time.time()
            phi_terms = compute_phi_terms(GT_Batch, sampler_te)
            phi_time = time.time() - start_time
            
            # Step 3: Compute energy
            print("Step 3: Computing stochastic_energy_tf...")
            start_time = time.time()
            energy, loc_energies = stochastic_energy_tf(psi_new, sampler_var, edge_pairs, GT_Batch, 0.0)
            energy_time = time.time() - start_time
            
            # Step 4: Compute gradients
            print("Step 4: Computing stochastic_gradients_tfv3...")
            start_time = time.time()
            stoch_loss, stoch_grads = stochastic_gradients_tfv3(phi_terms, GT_Batch, sampler_var)
            gradients_time = time.time() - start_time
            
            # Apply gradients
            optimizer.apply(stoch_grads, sampler_var.model.trainable_variables)
            
            # Post-iteration measurements
            post_iter_memory = process.memory_info().rss / (1024 * 1024)
            post_iter_tf_objects = count_tf_objects()
            
            # Record iteration metrics
            round_metrics['iterations'].append({
                'inner_iteration': inner_idx+1,
                'pre_memory': pre_iter_memory,
                'post_memory': post_iter_memory,
                'memory_diff': post_iter_memory - pre_iter_memory,
                'pre_tf_objects': pre_iter_tf_objects,
                'post_tf_objects': post_iter_tf_objects,
                'tf_objects_diff': post_iter_tf_objects - pre_iter_tf_objects,
                'mcmc_time': mcmc_time,
                'phi_time': phi_time,
                'energy_time': energy_time,
                'gradients_time': gradients_time,
                'total_time': mcmc_time + phi_time + energy_time + gradients_time
            })
            
            print(f"Iteration memory impact: {post_iter_memory - pre_iter_memory:.2f} MB")
            print(f"TF objects created: {post_iter_tf_objects - pre_iter_tf_objects}")
            
            # Clean up temporary variables to help with memory management
            del psi_new, phi_terms, energy, loc_energies, stoch_loss, stoch_grads
        
        # After all inner iterations, perform aggressive cleanup
        aggressive_memory_cleanup()
        
        # Post-round measurements
        post_round_memory = process.memory_info().rss / (1024 * 1024)
        post_round_tf_objects = count_tf_objects()
        post_round_tf_functions = inspect_tf_functions()
        
        # Update round metrics
        round_metrics.update({
            'post_memory': post_round_memory,
            'memory_growth': post_round_memory - pre_round_memory,
            'post_tf_objects': post_round_tf_objects,
            'tf_objects_growth': post_round_tf_objects - pre_round_tf_objects,
            'post_tf_functions': post_round_tf_functions,
            'tf_functions_growth': post_round_tf_functions - pre_round_tf_functions
        })
        
        # Add to overall metrics
        metrics.append(round_metrics)
        
        print(f"\nRound {round_idx+1} completed")
        print(f"Memory before round: {pre_round_memory:.2f} MB")
        print(f"Memory after round (post-cleanup): {post_round_memory:.2f} MB")
        print(f"Net memory growth: {post_round_memory - pre_round_memory:.2f} MB")
    
    # Plot the accumulation results
    plot_accumulation_results(metrics)
    
    return metrics

def plot_accumulation_results(metrics):
    """Create visualizations for memory accumulation test"""
    rounds = [m['round'] for m in metrics]
    
    # Create a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Memory growth across rounds
    plt.subplot(2, 2, 1)
    plt.plot(rounds, [m['pre_memory'] for m in metrics], 'b-', marker='o', label='Pre-Round')
    plt.plot(rounds, [m['post_memory'] for m in metrics], 'r-', marker='s', label='Post-Round')
    plt.title('Memory Before/After Rounds (MB)')
    plt.xlabel('Round')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Net memory growth per round
    plt.subplot(2, 2, 2)
    plt.bar(rounds, [m['memory_growth'] for m in metrics])
    plt.title('Net Memory Growth per Round (MB)')
    plt.xlabel('Round')
    plt.ylabel('Memory Growth (MB)')
    plt.grid(True)
    
    # Plot 3: Memory change within each round's iterations
    plt.subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
    for i, round_data in enumerate(metrics):
        inner_iterations = [iter_data['inner_iteration'] for iter_data in round_data['iterations']]
        memory_values = [iter_data['pre_memory'] for iter_data in round_data['iterations']]
        memory_values.append(round_data['iterations'][-1]['post_memory'])  # Add final post-memory
        extended_iterations = inner_iterations + [inner_iterations[-1] + 1]  # Add point for final memory
        plt.plot(extended_iterations, memory_values, 'o-', color=colors[i], 
                 label=f'Round {round_data["round"]}')
    
    plt.title('Memory Progression Within Rounds')
    plt.xlabel('Inner Iteration')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: TF objects growth within and across rounds
    plt.subplot(2, 2, 4)
    plt.plot(rounds, [m['pre_tf_objects'] for m in metrics], 'b-', marker='o', label='Pre-Round TF Objects')
    plt.plot(rounds, [m['post_tf_objects'] for m in metrics], 'r-', marker='s', label='Post-Round TF Objects')
    plt.plot(rounds, [m['pre_tf_functions'] for m in metrics], 'g--', marker='o', label='Pre-Round TF Functions')
    plt.plot(rounds, [m['post_tf_functions'] for m in metrics], 'y--', marker='s', label='Post-Round TF Functions')
    plt.title('TF Objects/Functions Growth')
    plt.xlabel('Round')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    import os
    os.makedirs('full_tracking', exist_ok=True)
    plt.savefig(f'full_tracking/memory_accumulation_{timestamp}.png')
    print(f"Saved plots to 'full_trackingmemory_accumulation_{timestamp}.png'")

if __name__ == "__main__":
    print("TensorFlow Memory Analysis Tool")
    print("------------------------------")
    print("1. Comprehensive operation tracking")
    print("2. Memory accumulation test")
    
    choice = input("Select analysis type (1-2): ")
    
    if choice == '1':
        # Run comprehensive operation tracking
        n_iter = int(input("Number of iterations (default 10): ") or 30)
        batch_size = int(input("Batch size (default 32): ") or 32)
        track_all_operations(n_iterations=n_iter, batch_size=batch_size)
    elif choice == '2':
        # Run memory accumulation test
        n_rounds = int(input("Number of rounds (default 3): ") or 3)
        n_inner = int(input("Inner iterations per round (default 5): ") or 5)
        batch_size = int(input("Batch size (default 32): ") or 32)
        run_accumulation_test(n_iterations=n_rounds, inner_iterations=n_inner, batch_size=batch_size)
    else:
        print("Invalid choice. Exiting.")