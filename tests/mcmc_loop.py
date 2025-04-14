import tensorflow as tf
import numpy as np
import gc
import psutil
import matplotlib.pyplot as plt
import os
import time
import weakref
from compgraph.monte_carlo import MCMCSampler
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt, precompute_graph_structure
# Set memory growth on GPUs if available - MUST BE DONE BEFORE ANY OTHER TF OPERATIONS
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

def detailed_memory_tracking(n_iterations=15, sweep_count=20, batch_size=32):
    """Track detailed memory usage across multiple Monte Carlo updates"""
    print("Setting up minimal environment...")
    
    # Force eager execution to prevent graph accumulation
    # print(f"Eager execution: {tf.executing_eagerly()}")
    
    # Create minimal graph and model
    graph_type = "2dsquare"
    lattice_size = (5, 5)
    sublattice = "Neel"
    beta = 0.1
    
    # Create graph and model with minimal parameters
    graph, subl = create_graph_from_ham(graph_type, lattice_size, sublattice)
    model_params = {"hidden_size": 32, "output_emb_size": 16}
    model = initialize_NQS_model_fromhyperparams("GNN2simple", model_params)
    
    # Create graph tuples and edge pairs
    GT_Batch = initialize_graph_tuples_tf_opt(batch_size, graph, subl)
    _, _, edge_pairs = precompute_graph_structure(graph)
    
    # Initialize model and sampler
    model(GT_Batch)  # Initialize weights
    sampler = MCMCSampler(model, GT_Batch, beta, edge_pairs=edge_pairs)
    
    # # Try to patch the monte_carlo_update_on_batch method to run in eager mode
    # if hasattr(sampler, 'monte_carlo_update_on_batch') and hasattr(sampler.monte_carlo_update_on_batch, 'python_function'):
    #     print("Attempting to patch monte_carlo_update_on_batch to run eagerly")
    #     original_fn = sampler.monte_carlo_update_on_batch.python_function
    #     setattr(sampler, 'monte_carlo_update_on_batch', original_fn)
    
    # Setup tracking
    process = psutil.Process()
    metrics = []
    
    # Force initial garbage collection
    aggressive_memory_cleanup()
    
    print("\nStarting memory tracking test...")
    print(f"Initial memory: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"Initial TF objects: {count_tf_objects()}")
    print(f"Initial TF functions: {inspect_tf_functions()}")
    
    # Run multiple monte_carlo_update_on_batch calls
    for i in range(n_iterations):
        # Pre-update memory and object counts
        aggressive_memory_cleanup()
        pre_memory = process.memory_info().rss / (1024 * 1024)
        pre_tf_objects = count_tf_objects()
        pre_tf_functions = inspect_tf_functions()
        
        start_time = time.time()
        
        # Store object ID instead of weak reference
        pre_gt_batch_id = id(GT_Batch)
        
        # Execute the target function
        try:
            with tf.profiler.experimental.Profile('memory_profile'):
                GT_Batch, psi_new = sampler.monte_carlo_update_on_batch(GT_Batch, sweep_count)
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            break
            
        # Immediately delete output to see if it helps
        psi_new_size = sum([t.numpy().nbytes for t in psi_new]) if isinstance(psi_new, list) else psi_new.numpy().nbytes
        print(f"  psi_new size: {psi_new_size / (1024 * 1024):.2f} MB")
        del psi_new
        
        # Post-update measurements
        execution_time = time.time() - start_time
        post_memory = process.memory_info().rss / (1024 * 1024)
        post_tf_objects = count_tf_objects()
        post_tf_functions = inspect_tf_functions()
        
        # Check if the GT_Batch reference is still alive
        if id(GT_Batch) == pre_gt_batch_id:
            print("  Warning: GT_Batch object hasn't been replaced (same id)")
        else:
            print("  GT_Batch was replaced with a new object")
        # Report GraphsTuple memory usage
        gt_size = sum([
            GT_Batch.nodes.numpy().nbytes if hasattr(GT_Batch.nodes, 'numpy') else 0,
            GT_Batch.edges.numpy().nbytes if hasattr(GT_Batch.edges, 'numpy') else 0,
            GT_Batch.globals.numpy().nbytes if hasattr(GT_Batch.globals, 'numpy') else 0
        ]) / (1024 * 1024)
        print(f"  GraphsTuple size: {gt_size:.2f} MB")
        
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
            'gt_size_mb': gt_size
        })
        
        print(f"Iteration {i+1}: Memory {pre_memory:.1f} → {post_memory:.1f} → {post_cleanup_memory:.1f} MB | "
              f"TF objects {pre_tf_objects} → {post_tf_objects} → {post_cleanup_tf_objects} | "
              f"TF functions {pre_tf_functions} → {post_tf_functions} → {post_cleanup_tf_functions} | "
              f"Time: {execution_time:.3f}s")
    
    # Visualize results
    plot_memory_results(metrics)
    
    # Print final results
    print("\nTest summary:")
    print(f"Total memory growth: {metrics[-1]['post_cleanup_memory'] - metrics[0]['pre_memory']:.2f} MB")
    print(f"TF object growth: {metrics[-1]['post_cleanup_tf_objects'] - metrics[0]['pre_tf_objects']}")
    print(f"TF function growth: {metrics[-1]['post_cleanup_tf_functions'] - metrics[0]['pre_tf_functions']}")
    
    return metrics

def plot_memory_results(metrics):
    """Generate plots from collected metrics"""
    iterations = [m['iteration'] for m in metrics]
    
    plt.figure(figsize=(15, 12))
    
    # Memory plot
    plt.subplot(3, 2, 1)
    plt.plot(iterations, [m['pre_memory'] for m in metrics], 'b-', label='Before')
    plt.plot(iterations, [m['post_memory'] for m in metrics], 'r-', label='After')
    plt.plot(iterations, [m['post_cleanup_memory'] for m in metrics], 'g--', label='After Cleanup')
    plt.title('Memory Usage (MB)')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # Memory diff plot
    plt.subplot(3, 2, 2)
    plt.bar(iterations, [m['memory_diff'] for m in metrics])
    plt.title('Memory Increase per Iteration (MB)')
    plt.xlabel('Iteration')
    plt.grid(True)
    
    # TF objects plot
    plt.subplot(3, 2, 3)
    plt.plot(iterations, [m['pre_tf_objects'] for m in metrics], 'b-', label='Before')
    plt.plot(iterations, [m['post_tf_objects'] for m in metrics], 'r-', label='After')
    plt.plot(iterations, [m['post_cleanup_tf_objects'] for m in metrics], 'g--', label='After Cleanup')
    plt.title('TensorFlow Objects Count')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # TF functions plot
    plt.subplot(3, 2, 4)
    plt.plot(iterations, [m['pre_tf_functions'] for m in metrics], 'b-', label='Before')
    plt.plot(iterations, [m['post_tf_functions'] for m in metrics], 'r-', label='After')
    plt.plot(iterations, [m['post_cleanup_tf_functions'] for m in metrics], 'g--', label='After Cleanup')
    plt.title('TensorFlow Functions Count')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # Execution time
    plt.subplot(3, 2, 5)
    plt.plot(iterations, [m['execution_time'] for m in metrics], 'k-')
    plt.title('Execution Time (seconds)')
    plt.xlabel('Iteration')
    plt.grid(True)
    
    # GraphsTuple size
    plt.subplot(3, 2, 6)
    plt.plot(iterations, [m['gt_size_mb'] for m in metrics], 'm-')
    plt.title('GraphsTuple Size (MB)')
    plt.xlabel('Iteration')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('memory_analysis_detailed.png')
    print("Saved plots to 'memory_analysis_detailed.png'")

if __name__ == "__main__":
    # Force TF to run in eager mode to test if that helps
    # tf.config.run_functions_eagerly(T    # # Try to patch the monte_carlo_update_on_batch method to run in eager mode
    # if hasattr(sampler, 'monte_carlo_update_on_batch') and hasattr(sampler.monte_carlo_update_on_batch, 'python_function'):
    #     printrue)
    # print("Running in eager execution mode:", tf.executing_eagerly())
    # tf.config.run_functions_eagerly(False)
    # Track and plot memory usage
    results = detailed_memory_tracking(n_iterations=15, sweep_count=40, batch_size=32)