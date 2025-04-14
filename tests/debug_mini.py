import tracemalloc
import memory_profiler
import functools
import time
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
from compgraph.tensor_wave_functions import montecarlo_logloss_overlap_time_evoluted, quimb_vec_to_sparse, variational_wave_function_on_batch, sparse_tensor_exp_energy, calculate_sparse_overlap,time_evoluted_wave_function_on_batch
from compgraph.monte_carlo import MCMCSampler
from memory_profiler import profile as mprofile

from compgraph.useful import graph_tuple_list_to_configs_list, copy_to_non_trainable, compare_sonnet_modules, sites_to_sparse_updated, create_amplitude_frequencies_from_graph_tuples,sparse_list_to_configs
from compgraph.monte_carlo import stochastic_energy
from simulation.initializer import create_graph_from_ham, format_hyperparams_to_string, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
import multiprocessing as mp
from multiprocessing import Pool

from compgraph.cg_repr import config_hamiltonian_product, graph_tuple_to_config_hamiltonian_product_update
from compgraph.useful import graph_tuple_toconfig, generate_graph_tuples_configs_tf

from memory_profiler import profile

from tensorflow.python.profiler import profiler_v2 as profiler_tf

from tests.debug_malloc import stochastic_gradients_malloc


def tracemalloc_snapshots(top_n=10, stop_tracing=True):
    """
    A decorator to take tracemalloc snapshots before and after
    the decorated function, then print the top memory allocations.

    Args:
        top_n (int): How many lines to display in the stats comparison.
        stop_tracing (bool): Whether to call tracemalloc.stop() after finishing.
                             If False, tracemalloc will keep running.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1) Start tracemalloc (if not already running)
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                print("[TRACEMALLOC] Started tracing.")

            # 2) Take a 'before' snapshot
            snapshot_before = tracemalloc.take_snapshot()

            # 3) Run the function
            result = func(*args, **kwargs)

            # 4) Take an 'after' snapshot
            snapshot_after = tracemalloc.take_snapshot()

            # 5) Compare snapshots
            stats_diff = snapshot_after.compare_to(snapshot_before, 'traceback')
            print("\n[TRACEMALLOC] Top allocations between snapshots:")
            for idx, stat in enumerate(stats_diff[:top_n], start=1):
                print(f"  {idx}. {stat}")

            # 6) Optionally stop tracing
            if stop_tracing:
                tracemalloc.stop()
                print("[TRACEMALLOC] Stopped tracing.\n")

            return result
        return wrapper
    return decorator


def start_tracemalloc():
    """
    Starts the tracemalloc module if not already started.
    Ideal to call once at the start of the script or before
    the suspected leak.
    """
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        print("[INFO] Tracemalloc started.")
    else:
        print("[INFO] Tracemalloc already running.")

def snapshot_tracemalloc(description=""):
    """
    Takes a snapshot and prints the top memory allocations (CPU).
    
    Args:
        description (str): A label/description for the snapshot.
    """
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('traceback')
    
    print(f"\n[TRACEMALLOC SNAPSHOT] {description}")
    for idx, stat in enumerate(top_stats[:10], start=1):
        print(f"  {idx}. {stat}")
    print("---------------------------------------------------------\n")

def compare_tracemalloc_snapshots(old_snapshot, new_snapshot, description=""):
    """
    Compares two snapshots to see which lines increased memory usage.
    
    Args:
        old_snapshot: The previous tracemalloc snapshot.
        new_snapshot: The current tracemalloc snapshot.
        description (str): A label/description for these snapshots.
    """
    stats_diff = new_snapshot.compare_to(old_snapshot, 'traceback')
    print(f"\n[TRACEMALLOC COMPARE] {description}")
    for idx, stat in enumerate(stats_diff[:10], start=1):
        print(f"  {idx}. {stat}")
    print("---------------------------------------------------------\n")

def measure_memory(func):
    """
    A decorator that uses memory_profiler.memory_usage() to check memory
    before and after the function call. This gives a quick overview of
    how much CPU memory the function uses.

    Returns:
        The result of the decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mem_before = memory_profiler.memory_usage()[0]
        t0 = time.time()
        
        result = func(*args, **kwargs)
        
        mem_after = memory_profiler.memory_usage()[0]
        t1 = time.time()
        
        used = mem_after - mem_before
        elapsed = t1 - t0
        
        print(f"[MEMORY PROFILER] Function: {func.__name__}")
        print(f"  Before: {mem_before:.2f} MB | After: {mem_after:.2f} MB | Used: {used:.2f} MB")
        print(f"  Elapsed Time: {elapsed:.2f} s\n")
        
        return result
    return wrapper

import gc

def mini_function(graph_tuples, model,i,amplitudes):
    gc.collect()
    n=50
    if i%n==0:
            snapshot_start=tracemalloc.take_snapshot()    
    for gt in graph_tuples:
        
        output=model(gt)
        numpy_output = output.numpy().copy()         
        # output=model(gt)[0]
        # amplitude, phase = output
        gelandomi_ami=tf.complex(real=numpy_output[0][0] * tf.cos(numpy_output[0][1] ),
                                     imag=numpy_output[0][0]   * tf.sin(numpy_output[0][1]))
        # tappost_frate=tf.complex(real=1 * tf.cos(1.5),imag=1 * tf.sin(1.5))
        # amplitudes.append(tappost_frate)
        print(numpy_output)
        del output, gelandomi_ami,numpy_output

        # del output
    if i%n==0:
        snapshot_after = tracemalloc.take_snapshot()
        top_stats_gt = snapshot_after.compare_to(snapshot_start, 'traceback')
        print(f"\n[TRACEMALLOC] application on new gt:")            
        for idx, stat in enumerate(top_stats_gt[:10], start=1):
                        print(f"  {idx}. {stat}")
    return                    

def main():
     # 1) Define your hyperparameters manually:
    hyperparams = {
        'simulation_type': 'VMC',       # Only using 'VMC' path, ignoring 'fsh'
        'graph_params': {
            'graphType': '2dsquare',
            'n': 3,
            'm': 3,
            'sublattice': 'Neel',
        },
        'sim_params': {
            'beta': 0.07,
            'batch_size': 4,
            'learning_rate': 7e-5,
            'outer_loop': 5,
            'inner_loop': 3,
        },
        'ansatz': 'GNN2simple',  # Name or type of your ansatz (adjust to your code)
        'ansatz_params': {
            'K_layer': None,
            'hidden_size': 128,
            'output_emb_size': 64,
        }
    }

    # 2) Create the graph and sublattice:
    graph, subl = create_graph_from_ham(
        hyperparams['graph_params']['graphType'],
        (hyperparams['graph_params']['n'], hyperparams['graph_params']['m']),
        hyperparams['graph_params']['sublattice']
    )

    # 3) If the total number of sites < 17, we can initialize
    #    a ground state for overlap comparisons:
    n_sites = hyperparams['graph_params']['n'] * hyperparams['graph_params']['m']
    if n_sites < 17:
        full_basis_configs = np.array([
            [int(x) for x in format(i, f'0{len(graph.nodes)}b')]
            for i in range(2**(len(graph.nodes)))
        ]) * 2 - 1
        lowest_eigenstate_as_sparse = initialize_hamiltonian_and_groundstate(
            hyperparams['graph_params'], full_basis_configs
        )
    else:
        lowest_eigenstate_as_sparse = None

    # 4) Initialize two models (trainable and fixed):
    model_w = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'],
                                                   hyperparams['ansatz_params'])
    model_fix = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'],
                                                     hyperparams['ansatz_params'])

    # 5) Create graph tuples for the variational (trainable) set and fixed set:
    graph_tuples_var = initialize_graph_tuples(hyperparams['sim_params']['batch_size'],
                                               graph, subl)
    graph_tuples_fix = initialize_graph_tuples(hyperparams['sim_params']['batch_size'],
                                               graph, subl)
    tracemalloc.start()
    gc.collect()
    # Add before your training loop
    profiler_tf.start('logs/tf_logs')
  
    amplitudes=[]
    for i in range(200):
        if i%2==0:
            graph_tuples=graph_tuples_var

            # mini_function(graph_tuples_var,model_w,i)
        else:
            graph_tuples=graph_tuples_fix        

            # mini_function(graph_tuples_fix,model_w,i)
        mini_function(graph_tuples,model_w,i,amplitudes)
    # Your training loop
    profiler_tf.stop()  
    return

if __name__ == "__main__":
    main()