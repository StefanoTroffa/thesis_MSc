import tracemalloc
import memory_profiler
import functools
import time

# If you want optional TensorFlow GPU memory checks, uncomment:
# import tensorflow as tf

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

# If you want to measure GPU memory usage in TensorFlow, you could do something like:
# def get_tf_gpu_memory(gpu_index=0):
#     """
#     For a quick GPU usage check with TF2,
#     returns a dict with 'current' and 'peak' usage in MB for the selected GPU.
#     """
#     gpus = tf.config.list_physical_devices('GPU')
#     if not gpus:
#         return {"current": 0, "peak": 0}
#
#     info = tf.config.experimental.get_memory_info(f'GPU:{gpu_index}')
#     # info["current"] and info["peak"] are in bytes; convert to MB
#     return {
#         "current": info["current"] / (1024**2),
#         "peak": info["peak"] / (1024**2)
#     }

def main():
    """
    Main function to demonstrate how to use both tracemalloc snapshots
    and the memory_profiler decorator in a single script.
    
    Steps:
      1) Start tracemalloc (CPU allocations).
      2) Take a baseline snapshot.
      3) Run a test function (decorated with measure_memory).
      4) Take a new snapshot, compare to the old one.
      5) (Optional) Check GPU memory usage if using TensorFlow.
    """
    # 1. Start tracemalloc
    start_tracemalloc()

    # 2. Take a baseline snapshot
    snapshot_before = tracemalloc.take_snapshot()

    # 3. Run a test function that might allocate memory
    test_function()

    # 4. Take a new snapshot and compare
    snapshot_after = tracemalloc.take_snapshot()
    snapshot_tracemalloc(description="After test_function")
    compare_tracemalloc_snapshots(snapshot_before, snapshot_after, 
                                  description="Baseline vs. after test_function")

    # 5. (Optional) Print GPU usage if using TensorFlow
    # gpu_usage = get_tf_gpu_memory()
    # print(f"GPU usage: Current={gpu_usage['current']:.2f}MB, Peak={gpu_usage['peak']:.2f}MB")





import time
import numpy as np
import tensorflow as tf
import quimb as qu
import sonnet as snt

from compgraph.tensor_wave_functions import montecarlo_logloss_overlap_time_evoluted, quimb_vec_to_sparse, variational_wave_function_on_batch, sparse_tensor_exp_energy, calculate_sparse_overlap,time_evoluted_wave_function_on_batch
from compgraph.monte_carlo import MCMCSampler
from memory_profiler import profile as mprofile
# from compgraph.monte_carlo import MCMCSampler
import line_profiler
import tracemalloc

from compgraph.useful import graph_tuple_list_to_configs_list, copy_to_non_trainable, compare_sonnet_modules, sites_to_sparse_updated, create_amplitude_frequencies_from_graph_tuples,sparse_list_to_configs
from compgraph.monte_carlo import stochastic_gradients_malloc, stochastic_energy
from simulation.initializer import create_graph_from_ham, format_hyperparams_to_string, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
import multiprocessing as mp
#from joblib import Parallel, delayed
from multiprocessing import Pool

from compgraph.cg_repr import config_hamiltonian_product, graph_tuple_to_config_hamiltonian_product_update
# import line_profiler
from compgraph.useful import graph_tuple_toconfig, generate_graph_tuples_configs_new

# @measure_memory

from memory_profiler import profile
import functools
import tracemalloc

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

# @tracemalloc_snapshots(top_n=15, stop_tracing=True)
import tracemalloc
import time
import numpy as np
import tensorflow as tf
import sonnet as snt

# ... other imports (MCMCSampler, etc.) ...

def outer_training_mc_2snap(outer_steps, inner_steps, graph,
                     beta, initial_learning_rate, model_w, model_fix, 
                     graph_tuples_var, graph_tuples_te, 
                     lowest_eigenstate_as_sparse=None):
    """
    Now with iteration-by-iteration tracemalloc snapshots for pinpointing
    repeated or incremental allocations.
    """

    # 1. Start Tracemalloc (only once, outside the loops)
    tracemalloc.start()
    print("[TRACEMALLOC] Started tracing in outer_training_mc.")

    n_sites = len(graph_tuples_var[0].nodes)
    N_sweeps = n_sites // 2

    start_time = time.time()
    energies = []
    loss_vectors = []
    overlap_in_time = []
    magnetizations = []

    # Model initialization
    initialize_model_w = model_w(graph_tuples_var[0])
    initialize_model_fix = model_fix(graph_tuples_te[0])

    # Initialize samplers
    sampler_var = MCMCSampler(model_w, graph_tuples_var[0])
    sampler_te = MCMCSampler(model_fix, graph_tuples_te[0], beta, graph)

    n_sites = len(graph_tuples_var[0].nodes[:, 0])
    optimizer = snt.optimizers.Adam(initial_learning_rate)

    # Precompute ground-state data if small enough system
    if n_sites < 17:
        fhs = np.array([[int(x) for x in format(i, f'0{n_sites}b')]
                        for i in range(2**(n_sites))]) * 2 - 1
        fh_gt = generate_graph_tuples_configs_new(graph_tuples_var[0], fhs)

    # 2. Take an initial "before" snapshot
    snapshot_before = tracemalloc.take_snapshot()

    # Outer loop
    for step in range(outer_steps):
        are_identical = compare_sonnet_modules(sampler_var.model, sampler_te.model)
        copy_to_non_trainable(sampler_var.model, sampler_te.model)
        are_identical2 = compare_sonnet_modules(sampler_var.model, sampler_te.model)

        # Inner loop
        for innerstep in range(inner_steps):
            # Monte Carlo updates
            graph_tuples_var, coeff_var_on_var = zip(*[
                sampler_var.monte_carlo_update(N_sweeps, gt, 'var') 
                for gt in graph_tuples_var
            ])

            configs_var = sites_to_sparse_updated(
                graph_tuple_list_to_configs_list(graph_tuples_var)
            )

            wave_function_var_on_var, freq_var = create_amplitude_frequencies_from_graph_tuples(
                graph_tuples_var, coeff_var_on_var
            )
            freq_ampl_var = np.array(freq_var.values) / len(graph_tuples_var)
            unique_tuples_var = generate_graph_tuples_configs_new(
                graph_tuples_var[0], 
                sparse_list_to_configs(freq_var.indices[:, 0], n_sites)
            )

            stoch_gradients = stochastic_gradients(
                sampler_var, sampler_te, unique_tuples_var, freq_ampl_var
            )
            optimizer.apply(stoch_gradients, sampler_var.model.trainable_variables)
            stoch_energy = stochastic_energy(sampler_var, graph, unique_tuples_var, freq_ampl_var)

            energies.append(stoch_energy[0].numpy())

            # Compute magnetization
            def graph_tuple_toconfig_tf(graph_tuple):
                config = graph_tuple.nodes[:, 0]
                return config

            configs = [graph_tuple_toconfig_tf(sample) for sample in unique_tuples_var]
            configs_tensor = tf.stack(configs)
            Sz_s = tf.reduce_sum(configs_tensor, axis=1)
            freq_ampl_var_tensor = tf.convert_to_tensor(freq_ampl_var, dtype=Sz_s.dtype)

            total_Sz = tf.reduce_sum(Sz_s * freq_ampl_var_tensor)
            total_frequency = tf.reduce_sum(freq_ampl_var_tensor)

            N = tf.cast(tf.shape(configs_tensor)[1], Sz_s.dtype)
            M_z = total_Sz / total_frequency
            m_z = M_z / N
            magnetizations.append(m_z.numpy())

        # (Optional) do something else each outer step,
        # e.g. evaluate overlap if n_sites < 17
        if n_sites < 17:
            outputs = variational_wave_function_on_batch(sampler_var.model, fh_gt)
            normaliz_gnn = 1 / tf.norm(outputs.values)
            norm_low_state_gnn = tf.sparse.map_values(tf.multiply, outputs, normaliz_gnn)
            overlap_temp = tf.norm(calculate_sparse_overlap(
                lowest_eigenstate_as_sparse, norm_low_state_gnn
            ))
            overlap_in_time.append(overlap_temp.numpy())

        # 3. Take an "after" snapshot at the end of each outer step
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'traceback')
        print(f"\n[TRACEMALLOC] Memory usage after outer step {step}:")
        for idx, stat in enumerate(top_stats[:10], start=1):
            print(f"  {idx}. {stat}")

        # Update the "before" snapshot so that next iteration we see incremental changes
        snapshot_before = snapshot_after

    endtime = time.time() - start_time

    # 4. (Optional) stop tracemalloc
    tracemalloc.stop()
    print("[TRACEMALLOC] Stopped tracing in outer_training_mc.")

    return endtime, energies, loss_vectors, overlap_in_time, magnetizations

import numpy as np
import tensorflow as tf
def outer_training_mc(outer_steps, inner_steps, graph,
                      beta, initial_learning_rate, model_w, model_fix, 
                      graph_tuples_var, graph_tuples_te, 
                      lowest_eigenstate_as_sparse=None):
    """
    Outer training loop with additional tracemalloc snapshots inserted after key steps:
      - After the Monte Carlo update.
      - After computing and applying gradients.
    """


    # 1. Start Tracemalloc (only once, outside the loops)
    tracemalloc.start()
    print("[TRACEMALLOC] Started tracing in outer_training_mc.")

    n_sites = len(graph_tuples_var[0].nodes)
    N_sweeps = n_sites // 2

    start_time = time.time()
    energies = []
    loss_vectors = []
    overlap_in_time = []
    magnetizations = []

    # Model initialization
    initialize_model_w = model_w(graph_tuples_var[0])
    initialize_model_fix = model_fix(graph_tuples_te[0])

    # Initialize samplers
    sampler_var = MCMCSampler(model_w, graph_tuples_var[0])
    sampler_te = MCMCSampler(model_fix, graph_tuples_te[0], beta, graph)

    n_sites = len(graph_tuples_var[0].nodes[:, 0])
    optimizer = snt.optimizers.Adam(initial_learning_rate)

    # Precompute ground-state data if small enough system
    if n_sites < 17:
        fhs = np.array([[int(x) for x in format(i, f'0{n_sites}b')]
                        for i in range(2**(n_sites))]) * 2 - 1
        fh_gt = generate_graph_tuples_configs_new(graph_tuples_var[0], fhs)

    # 2. Take an initial "before" snapshot
    snapshot_before = tracemalloc.take_snapshot()
    import gc
    # Outer loop
    for step in range(outer_steps):
        are_identical = compare_sonnet_modules(sampler_var.model, sampler_te.model)
        copy_to_non_trainable(sampler_var.model, sampler_te.model)
        are_identical2 = compare_sonnet_modules(sampler_var.model, sampler_te.model)
        tf.keras.backend.clear_session()
        gc.collect()

        # Inner loop
        for innerstep in range(inner_steps):
            # --- Monte Carlo Updates ---
            snapshot_inner_before = tracemalloc.take_snapshot()
            
            # Monte Carlo updates (this returns a tuple for each element)
            graph_tuples_var, coeff_var_on_var = zip(*[
                sampler_var.monte_carlo_update(N_sweeps, gt, 'var') 
                for gt in graph_tuples_var
            ])
            
            # Take a snapshot after MC updates
            snapshot_after_mc = tracemalloc.take_snapshot()
            top_stats_mc = snapshot_after_mc.compare_to(snapshot_inner_before, 'traceback')
            print(f"\n[TRACEMALLOC] After Monte Carlo update (outer step {step}, inner step {innerstep}):")
            for idx, stat in enumerate(top_stats_mc[:5], start=1):
                print(f"  {idx}. {stat}")
            
            configs_var = sites_to_sparse_updated(
                graph_tuple_list_to_configs_list(graph_tuples_var)
            )

            wave_function_var_on_var, freq_var = create_amplitude_frequencies_from_graph_tuples(
                graph_tuples_var, coeff_var_on_var
            )
            freq_ampl_var = np.array(freq_var.values) / len(graph_tuples_var)
            unique_tuples_var = generate_graph_tuples_configs_new(
                graph_tuples_var[0], 
                sparse_list_to_configs(freq_var.indices[:, 0], n_sites)
            )

            # --- Compute and Apply Gradients ---
            snapshot_grad_before = tracemalloc.take_snapshot()
            
            stoch_gradients = stochastic_gradients_malloc(
                sampler_var, sampler_te, unique_tuples_var, freq_ampl_var
            )
            optimizer.apply(stoch_gradients, sampler_var.model.trainable_variables)
            
            # Take a snapshot after applying gradients
            snapshot_after_grad = tracemalloc.take_snapshot()
            top_stats_grad = snapshot_after_grad.compare_to(snapshot_grad_before, 'traceback')
            print(f"\n[TRACEMALLOC] After gradient computation (outer step {step}, inner step {innerstep}):")
            for idx, stat in enumerate(top_stats_grad[:5], start=1):
                print(f"  {idx}. {stat}")
            
            stoch_energy = stochastic_energy(sampler_var, graph, unique_tuples_var, freq_ampl_var)
            energies.append(stoch_energy[0].numpy())

            # --- Compute Magnetization ---
            snapshot_mag_before = tracemalloc.take_snapshot()
            
            def graph_tuple_toconfig_tf(graph_tuple):
                config = graph_tuple.nodes[:, 0]
                return config

            configs = [graph_tuple_toconfig_tf(sample) for sample in unique_tuples_var]
            configs_tensor = tf.stack(configs)
            Sz_s = tf.reduce_sum(configs_tensor, axis=1)
            freq_ampl_var_tensor = tf.convert_to_tensor(freq_ampl_var, dtype=Sz_s.dtype)

            total_Sz = tf.reduce_sum(Sz_s * freq_ampl_var_tensor)
            total_frequency = tf.reduce_sum(freq_ampl_var_tensor)

            N = tf.cast(tf.shape(configs_tensor)[1], Sz_s.dtype)
            M_z = total_Sz / total_frequency
            m_z = M_z / N
            magnetizations.append(m_z.numpy())
            
            snapshot_mag_after = tracemalloc.take_snapshot()
            top_stats_mag = snapshot_mag_after.compare_to(snapshot_mag_before, 'traceback')
            print(f"\n[TRACEMALLOC] After magnetization calculation (outer step {step}, inner step {innerstep}):")
            for idx, stat in enumerate(top_stats_mag[:5], start=1):
                print(f"  {idx}. {stat}")

        # (Optional) evaluate overlap if n_sites < 17 at the end of outer step
        if n_sites < 17:
            outputs = variational_wave_function_on_batch(sampler_var.model, fh_gt)
            normaliz_gnn = 1 / tf.norm(outputs.values)
            norm_low_state_gnn = tf.sparse.map_values(tf.multiply, outputs, normaliz_gnn)
            overlap_temp = tf.norm(calculate_sparse_overlap(
                lowest_eigenstate_as_sparse, norm_low_state_gnn
            ))
            overlap_in_time.append(overlap_temp.numpy())

        # 3. Take an "after" snapshot at the end of each outer step
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'traceback')
        print(f"\n[TRACEMALLOC] Memory usage after outer step {step}:")
        for idx, stat in enumerate(top_stats[:10], start=1):
            print(f"  {idx}. {stat}")

        # Update the "before" snapshot so that next iteration we see incremental changes
        snapshot_before = snapshot_after

    endtime = time.time() - start_time

    # 4. Stop tracemalloc
    tracemalloc.stop()
    print("[TRACEMALLOC] Stopped tracing in outer_training_mc.")

    return endtime, energies, loss_vectors, overlap_in_time, magnetizations


# NOTE:
# If for some reason you don't have these exact imports accessible,
# copy those functions or fix the import paths as appropriate.

def minimal_vmc_run():
    """
    Runs a minimal example of the outer_training_mc function with
    fixed hyperparameters. Adapts only the essentials to get a result.
    """

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
            'outer_loop': 40,
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

    # 6) Call the outer_training_mc function:
    endtime, energies, loss_vectors, overlap_in_time, magnetizations = outer_training_mc(
        outer_steps=hyperparams['sim_params']['outer_loop'],
        inner_steps=hyperparams['sim_params']['inner_loop'],
        graph=graph,
        beta=hyperparams['sim_params']['beta'],
        initial_learning_rate=hyperparams['sim_params']['learning_rate'],
        model_w=model_w,
        model_fix=model_fix,
        graph_tuples_var=graph_tuples_var,
        graph_tuples_te=graph_tuples_fix,
        lowest_eigenstate_as_sparse=lowest_eigenstate_as_sparse
    )

    # 7) Print a brief summary of results:
    print("\n========= Simulation Summary =========")
    print(f"Total simulation time: {endtime} seconds")
    print(f"Last energy in time: {energies[-1] if energies else None}")
    print(f"Last overlap in time: {overlap_in_time[-1] if overlap_in_time else None}")
    print(f"Last magnetization in time: {magnetizations[-1] if magnetizations else None}")
    print("======================================\n")


if __name__ == "__main__":
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)    
    minimal_vmc_run()

