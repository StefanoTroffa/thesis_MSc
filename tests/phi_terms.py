import tensorflow as tf
import numpy as np
import gc
import psutil
import matplotlib.pyplot as plt
import os
import time
import weakref
from compgraph.monte_carlo import MCMCSampler, compute_phi_terms
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure
from compgraph.useful import copy_to_non_trainable
import sonnet as snt

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
def count_tf_objects():
    """Count TensorFlow objects in memory"""
    tensors = [obj for obj in gc.get_objects() 
               if isinstance(obj, tf.Tensor) or isinstance(obj, tf.Variable)]
    return len(tensors)

# Function to inspect internal TensorFlow graphs
def inspect_tf_functions():
    """Inspect compiled TensorFlow functions"""
    concrete_functions = []
    for obj in gc.get_objects():
        if isinstance(obj, tf.types.experimental.ConcreteFunction):
            concrete_functions.append(obj)
    return len(concrete_functions)

def aggressive_memory_cleanup():
    """Aggressive memory cleanup to free TensorFlow resources"""
    # Clear Python objects
    gc.collect()
    
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Force TensorFlow to release memory back to the system
    if tf.config.list_physical_devices('GPU'):
        # For GPU - try to clear GPU memory
        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            print(f"Before GPU cleanup: {gpu_memory['current'] / 1024 / 1024:.1f} MB")
        except:
            pass
        
        # Reset GPU memory stats
        tf.config.experimental.reset_memory_stats('GPU:0')
        
        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            print(f"After GPU cleanup: {gpu_memory['current'] / 1024 / 1024:.1f} MB")
        except:
            pass

def track_phi_terms_memory(n_iterations=15, batch_size=32):
    """Track detailed memory usage across multiple phi_terms computations"""
    print("Setting up minimal environment...")
    
    # Force eager execution to prevent graph accumulation
    # print(f"Eager execution: {tf.executing_eagerly()}")
    
    # Create minimal graph and model
    graph_type = "2dsquare"
    lattice_size = (3, 3)
    sublattice = "Neel"
    beta = 0.1
    
    # Create graph and model with minimal parameters
    graph, subl = create_graph_from_ham(graph_type, lattice_size, sublattice)
    model_params = {"hidden_size": 32, "output_emb_size": 16}
    
    # Initialize two models - one for sampler_var and one for sampler_te
    model_w = initialize_NQS_model_fromhyperparams("GNN2simple", model_params)
    model_fix = initialize_NQS_model_fromhyperparams("GNN2simple", model_params)
    
    # Create graph tuples and edge pairs
    GT_Batch = initialize_graph_tuples_tf_opt(batch_size, graph, subl)
    senders, receivers, edge_pairs = precompute_graph_structure(graph)
    
    # Initialize models
    model_w(GT_Batch)  # Initialize weights
    model_fix(GT_Batch)  # Initialize weights
    
    # Copy weights from model_w to model_fix
    copy_to_non_trainable(model_w, model_fix)
    
    # Create samplers
    sampler_var = MCMCSampler(model_w, GT_Batch, beta, edge_pairs=edge_pairs)
    sampler_te = MCMCSampler(model_fix, GT_Batch, beta, edge_pairs=edge_pairs)
    
    # Setup tracking
    process = psutil.Process()
    metrics = []
    
    # Force initial garbage collection
    aggressive_memory_cleanup()
    
    print("\nStarting phi_terms memory tracking test...")
    print(f"Initial memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"Initial TF objects: {count_tf_objects()}")
    print(f"Initial TF functions: {inspect_tf_functions()}")
    
    # Run multiple phi_terms computations
    for i in range(n_iterations):
        # Pre-computation memory and object counts
        aggressive_memory_cleanup()
        pre_memory = process.memory_info().rss / (1024 * 1024)
        pre_tf_objects = count_tf_objects()
        pre_tf_functions = inspect_tf_functions()
        
        # First update the batch with Monte Carlo sampling
        print("First running monte_carlo_update_on_batch to generate valid configurations...")
        GT_Batch, _ = sampler_var.monte_carlo_update_on_batch(GT_Batch, 20)
        
        start_time = time.time()
        
        # Execute compute_phi_terms
        try:
            with tf.profiler.experimental.Profile('phi_terms_profile'):
                phi_terms = compute_phi_terms(GT_Batch, sampler_te)
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            break
            
        # Immediately measure phi_terms size
        phi_terms_size = tf.size(phi_terms).numpy() * phi_terms.dtype.size
        print(f"  phi_terms size: {phi_terms_size / (1024 * 1024):.2f} MB")
        
        # Explicitly delete to help garbage collection
        del phi_terms
        
        # Post-computation measurements
        execution_time = time.time() - start_time
        post_memory = process.memory_info().rss / (1024 * 1024)
        post_tf_objects = count_tf_objects()
        post_tf_functions = inspect_tf_functions()
        
        # Force cleanup
        aggressive_memory_cleanup()
        
        # After cleanup measurements
        post_cleanup_memory = process.memory_info().rss / (1024 * 1024)
        post_cleanup_tf_objects = count_tf_objects()
        post_cleanup_tf_functions = inspect_tf_functions()
        
        # Record metrics
        metrics.append({
            'iteration': i+1,
            'pre_memory': pre_memory,
            'post_memory': post_memory,
            'post_cleanup_memory': post_cleanup_memory,
            'memory_diff': post_memory - pre_memory,
            'pre_tf_objects': pre_tf_objects,
            'post_tf_objects': post_tf_objects, 
            'post_cleanup_tf_objects': post_cleanup_tf_objects,
            'tf_objects_diff': post_tf_objects - pre_tf_objects,
            'pre_tf_functions': pre_tf_functions,
            'post_tf_functions': post_tf_functions,
            'post_cleanup_tf_functions': post_cleanup_tf_functions,
            'execution_time': execution_time,
        })
        
        print(f"Iteration {i+1}: Memory {pre_memory:.1f} → {post_memory:.1f} → {post_cleanup_memory:.1f} MB | "
              f"TF objects {pre_tf_objects} → {post_tf_objects} → {post_cleanup_tf_objects} | "
              f"TF functions {pre_tf_functions} → {post_tf_functions} → {post_cleanup_tf_functions} | "
              f"Time: {execution_time:.3f}s")
    
    # Visualize results
    plot_phi_terms_memory_results(metrics)
    
    # Print final results
    print("\nTest summary:")
    print(f"Total memory growth: {metrics[-1]['post_cleanup_memory'] - metrics[0]['pre_memory']:.2f} MB")
    print(f"TF object growth: {metrics[-1]['post_cleanup_tf_objects'] - metrics[0]['pre_tf_objects']}")
    print(f"TF function growth: {metrics[-1]['post_cleanup_tf_functions'] - metrics[0]['pre_tf_functions']}")
    
    return metrics

def plot_phi_terms_memory_results(metrics):
    """Generate plots from collected metrics for phi_terms computation"""
    iterations = [m['iteration'] for m in metrics]
    
    plt.figure(figsize=(15, 10))
    
    # Memory plot
    plt.subplot(2, 2, 1)
    plt.plot(iterations, [m['pre_memory'] for m in metrics], 'b-', label='Before')
    plt.plot(iterations, [m['post_memory'] for m in metrics], 'r-', label='After')
    plt.plot(iterations, [m['post_cleanup_memory'] for m in metrics], 'g--', label='After Cleanup')
    plt.title('Memory Usage During compute_phi_terms (MB)')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # Memory diff plot
    plt.subplot(2, 2, 2)
    plt.bar(iterations, [m['memory_diff'] for m in metrics])
    plt.title('Memory Increase per Iteration (MB)')
    plt.xlabel('Iteration')
    plt.grid(True)
    
    # TF objects plot
    plt.subplot(2, 2, 3)
    plt.plot(iterations, [m['pre_tf_objects'] for m in metrics], 'b-', label='Before')
    plt.plot(iterations, [m['post_tf_objects'] for m in metrics], 'r-', label='After')
    plt.plot(iterations, [m['post_cleanup_tf_objects'] for m in metrics], 'g--', label='After Cleanup')
    plt.title('TensorFlow Objects Count')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # Execution time
    plt.subplot(2, 2, 4)
    plt.plot(iterations, [m['execution_time'] for m in metrics], 'k-')
    plt.title('Execution Time (seconds)')
    plt.xlabel('Iteration')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('phi_terms_memory_analysis.png')
    print("Saved plots to 'phi_terms_memory_analysis.png'")

def create_combined_tracking_experiment(n_iterations=10, batch_size=32):
    """Run a comprehensive experiment that tracks both MCMC and phi_terms memory usage"""
    print("Setting up minimal environment for combined tracking...")
    
    # Create minimal graph and model
    graph_type = "2dsquare"
    lattice_size = (3, 3)
    sublattice = "Neel"
    beta = 0.1
    
    # Create graph and model with minimal parameters
    graph, subl = create_graph_from_ham(graph_type, lattice_size, sublattice)
    model_params = {"hidden_size": 32, "output_emb_size": 16}
    
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
    
    # Setup tracking
    process = psutil.Process()
    metrics = []
    
    # Force initial garbage collection
    aggressive_memory_cleanup()
    
    print("\nStarting combined tracking experiment...")
    print(f"Initial memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    
    for i in range(n_iterations):
        print(f"\nIteration {i+1}/{n_iterations}")
        aggressive_memory_cleanup()
        
        # Pre-operations measurements
        pre_memory = process.memory_info().rss / (1024 * 1024)
        pre_tf_objects = count_tf_objects()
        
        # Step 1: Run monte_carlo_update_on_batch
        print("Step 1: Running monte_carlo_update_on_batch...")
        start_time = time.time()
        GT_Batch, psi_new = sampler_var.monte_carlo_update_on_batch(GT_Batch, 20)
        mcmc_time = time.time() - start_time
        
        # Post-MCMC measurements
        post_mcmc_memory = process.memory_info().rss / (1024 * 1024)
        post_mcmc_tf_objects = count_tf_objects()
        
        # Delete psi_new to help garbage collection
        del psi_new
        
        # Step 2: Compute phi_terms
        print("Step 2: Computing phi_terms...")
        start_time = time.time()
        phi_terms = compute_phi_terms(GT_Batch, sampler_te)
        phi_terms_time = time.time() - start_time
        
        # Post-phi_terms measurements
        post_phi_memory = process.memory_info().rss / (1024 * 1024)
        post_phi_tf_objects = count_tf_objects()
        
        # Delete phi_terms to help garbage collection
        del phi_terms
        
        # Force cleanup
        aggressive_memory_cleanup()
        
        # After cleanup measurements
        post_cleanup_memory = process.memory_info().rss / (1024 * 1024)
        post_cleanup_tf_objects = count_tf_objects()
        
        # Record metrics
        metrics.append({
            'iteration': i+1,
            'pre_memory': pre_memory,
            'post_mcmc_memory': post_mcmc_memory,
            'post_phi_memory': post_phi_memory,
            'post_cleanup_memory': post_cleanup_memory,
            'mcmc_memory_diff': post_mcmc_memory - pre_memory,
            'phi_memory_diff': post_phi_memory - post_mcmc_memory,
            'pre_tf_objects': pre_tf_objects,
            'post_mcmc_tf_objects': post_mcmc_tf_objects,
            'post_phi_tf_objects': post_phi_tf_objects, 
            'post_cleanup_tf_objects': post_cleanup_tf_objects,
            'mcmc_time': mcmc_time,
            'phi_time': phi_terms_time,
        })
        
        print(f"Memory: {pre_memory:.1f} → {post_mcmc_memory:.1f} → {post_phi_memory:.1f} → {post_cleanup_memory:.1f} MB")
        print(f"TF objects: {pre_tf_objects} → {post_mcmc_tf_objects} → {post_phi_tf_objects} → {post_cleanup_tf_objects}")
    
    # Plot combined results
    plot_combined_results(metrics)
    
    return metrics

def plot_combined_results(metrics):
    """Generate plots comparing MCMC and phi_terms memory usage"""
    iterations = [m['iteration'] for m in metrics]
    
    plt.figure(figsize=(15, 15))
    
    # Memory usage across stages
    plt.subplot(3, 2, 1)
    plt.plot(iterations, [m['pre_memory'] for m in metrics], 'b-', label='Before')
    plt.plot(iterations, [m['post_mcmc_memory'] for m in metrics], 'g-', label='After MCMC')
    plt.plot(iterations, [m['post_phi_memory'] for m in metrics], 'r-', label='After phi_terms')
    plt.plot(iterations, [m['post_cleanup_memory'] for m in metrics], 'k--', label='After Cleanup')
    plt.title('Memory Usage Across Processing Stages (MB)')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # Memory diff comparison
    plt.subplot(3, 2, 2)
    width = 0.35
    plt.bar(np.array(iterations) - width/2, [m['mcmc_memory_diff'] for m in metrics], width, label='MCMC Memory Impact')
    plt.bar(np.array(iterations) + width/2, [m['phi_memory_diff'] for m in metrics], width, label='phi_terms Memory Impact')
    plt.title('Memory Increase Comparison (MB)')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # TF objects across stages
    plt.subplot(3, 2, 3)
    plt.plot(iterations, [m['pre_tf_objects'] for m in metrics], 'b-', label='Before')
    plt.plot(iterations, [m['post_mcmc_tf_objects'] for m in metrics], 'g-', label='After MCMC')
    plt.plot(iterations, [m['post_phi_tf_objects'] for m in metrics], 'r-', label='After phi_terms')
    plt.plot(iterations, [m['post_cleanup_tf_objects'] for m in metrics], 'k--', label='After Cleanup')
    plt.title('TensorFlow Objects Count')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # TF object increase comparison
    plt.subplot(3, 2, 4)
    width = 0.35
    plt.bar(np.array(iterations) - width/2, 
            [m['post_mcmc_tf_objects'] - m['pre_tf_objects'] for m in metrics], 
            width, label='MCMC Object Impact')
    plt.bar(np.array(iterations) + width/2, 
            [m['post_phi_tf_objects'] - m['post_mcmc_tf_objects'] for m in metrics], 
            width, label='phi_terms Object Impact')
    plt.title('TF Object Count Increase Comparison')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # Execution time comparison
    plt.subplot(3, 2, 5)
    width = 0.35
    plt.bar(np.array(iterations) - width/2, [m['mcmc_time'] for m in metrics], width, label='MCMC Time (s)')
    plt.bar(np.array(iterations) + width/2, [m['phi_time'] for m in metrics], width, label='phi_terms Time (s)')
    plt.title('Execution Time Comparison (seconds)')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # Memory leak detection - post cleanup trend
    plt.subplot(3, 2, 6)
    plt.plot(iterations, [m['post_cleanup_memory'] for m in metrics], 'r-', marker='o')
    plt.title('Post-Cleanup Memory (Leak Detection)')
    plt.xlabel('Iteration')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('combined_memory_analysis.png')
    print("Saved plots to 'combined_memory_analysis.png'")

if __name__ == "__main__":
    # Choose the type of analysis to run
    analysis_type = "phi_terms"  # Options: "phi_terms", "combined"
    
    if analysis_type == "phi_terms":
        results = track_phi_terms_memory(n_iterations=15, batch_size=32)
    else:
        results = create_combined_tracking_experiment(n_iterations=60, batch_size=64)