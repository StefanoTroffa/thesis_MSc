import time
import numpy as np
import tensorflow as tf
import sonnet as snt
import psutil
import os
from datetime import datetime
from memory_profiler import profile
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, calculate_sparse_overlap
from compgraph.monte_carlo import MCMCSampler
from compgraph.useful import copy_to_non_trainable, sites_to_sparse_updated, create_amplitude_frequencies_from_graph_tuples, sparse_list_to_configs
from compgraph.monte_carlo import stochastic_energy
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
from compgraph.useful import generate_graph_tuples_configs_tf
from dataclasses import dataclass, field
from compgraph.useful import graph_tuple_list_to_configs_list, copy_to_non_trainable, compare_sonnet_modules, sites_to_sparse_updated, create_amplitude_frequencies_from_graph_tuples,sparse_list_to_configs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0 = all messages are logged (default is 2)
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
# tf.autograph.set_verbosity(2)
# --- Helper Functions ---
def setup_tensorboard_logging():
    """Set up TensorBoard logging with a unique directory for each run"""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/vmc_run_{current_time}"
    # log_dir = "logs/vmc_run"
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer, log_dir

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

# @profile

# @tf.function()
def stochastic_gradients_malloc(sampler_var, sampler_te, unique_tuples_var, freq_ampl_var):
    """Compute stochastic gradients with memory profiling."""
    
    phi_terms= tf.stack([sampler_te.time_evoluted_config_amplitude(gt) for gt in unique_tuples_var])

    with tf.GradientTape() as tape:
        tape.watch(sampler_var.model.trainable_variables)
        print("Executing eagerly?", tf.executing_eagerly())  # Should be False inside @tf.function

        psi_terms= tf.stack([sampler_var.evaluate_model(gt) for gt in unique_tuples_var])

        log_psi = tf.math.log(tf.math.conj(psi_terms))
        ratio_phi_psi = tf.stop_gradient(phi_terms / psi_terms)
        stoch_estimation = freq_ampl_var * log_psi - ratio_phi_psi * log_psi * freq_ampl_var / tf.reduce_sum(ratio_phi_psi * freq_ampl_var)
        loss = tf.reduce_sum(stoch_estimation)

    stoch_gradients = tape.gradient(loss, sampler_var.model.trainable_variables)
    return stoch_gradients
# @profile
def outer_training_mc(outer_steps, inner_steps, graph, beta, initial_learning_rate, model_w, model_fix, 
                     graph_tuples_var, graph_tuples_te, lowest_eigenstate_as_sparse=None):
    """Outer training loop with memory profiling and TensorBoard logging."""
    summary_writer, log_dir = setup_tensorboard_logging()

    print("\nTensorBoard logs written to:", log_dir)
    print("To view the results, run:")
    print(f"tensorboard --logdir {log_dir}")
    # tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if physical_devices:
    #     for device in physical_devices:
    #         tf.config.experimental.set_memory_growth(device, True)
    #     print(f"GPU detected: {physical_devices}")
    physical_devices = tf.config.list_physical_devices('GPU')
    n_sites = len(graph_tuples_var[0].nodes)
    N_sweeps = n_sites // 2

    start_time = time.time()
    energies = []
    loss_vectors = []
    overlap_in_time = []
    magnetizations = []

    initialize_model_w = model_w(graph_tuples_var[0])
    initialize_model_fix = model_fix(graph_tuples_te[0])

    sampler_var = MCMCSampler(model_w, graph_tuples_var[0])
    sampler_te = MCMCSampler(model_fix, graph_tuples_te[0], beta, graph)

    n_sites = len(graph_tuples_var[0].nodes[:, 0])
    optimizer = snt.optimizers.Adam(initial_learning_rate)
    
    if n_sites < 17:
        fhs = np.array([[int(x) for x in format(i, f'0{n_sites}b')] for i in range(2**(n_sites))]) * 2 - 1
        fh_gt = generate_graph_tuples_configs_tf(graph_tuples_var[0], fhs)

    with summary_writer.as_default():
        tf.summary.text('configuration/hyperparameters', f'beta: {beta}\nlearning_rate: {initial_learning_rate}\n', step=0)

        for step in range(outer_steps):
            metrics = {
                'energy': energies[-1] if energies else None,
                'magnetization': magnetizations[-1] if magnetizations else None,
                'overlap': overlap_in_time[-1] if overlap_in_time else None,
                'learning_rate': initial_learning_rate,
                'ram_used_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'notes': f"Outer step {step} completed"
            }

            if physical_devices:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                metrics.update({
                    'gpu_memory_mb': gpu_memory['current'] / (1024 * 1024),
                })

            if step > 0:
                log_training_metrics(summary_writer, step, metrics)

            copy_to_non_trainable(sampler_var.model, sampler_te.model)
            tf.keras.backend.clear_session()

            # @tf.function
            def monte_carlo_step(graph_tuple):
                return sampler_var.monte_carlo_update(N_sweeps, graph_tuple, 'var')

            for innerstep in range(inner_steps):
                graph_tuples_var, coeff_var_on_var = zip(*[monte_carlo_step(gt) for gt in graph_tuples_var])
                configs_var = sites_to_sparse_updated(graph_tuple_list_to_configs_list(graph_tuples_var))

                wave_function_var_on_var, freq_var = create_amplitude_frequencies_from_graph_tuples(graph_tuples_var, coeff_var_on_var)
                freq_ampl_var = np.array(freq_var.values) / len(graph_tuples_var)
                unique_tuples_var = generate_graph_tuples_configs_tf(graph_tuples_var[0], sparse_list_to_configs(freq_var.indices[:, 0], n_sites))
                tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)
                stoch_gradients = stochastic_gradients_malloc(sampler_var, sampler_te, unique_tuples_var, freq_ampl_var)
                with summary_writer.as_default():
                    tf.summary.trace_export(name="stochastic_gradients_trace", step=0,)
                    print(tf.autograph.to_code(stochastic_gradients_malloc.python_function))

                optimizer.apply(stoch_gradients, sampler_var.model.trainable_variables)
                stoch_energy = stochastic_energy(sampler_var, graph, unique_tuples_var, freq_ampl_var)
                energies.append(stoch_energy[0].numpy())
                del stoch_energy, stoch_gradients
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
                magnetizations.append(float(m_z.numpy()))

            if n_sites < 17:
                outputs = variational_wave_function_on_batch(sampler_var.model, fh_gt)
                normaliz_gnn = 1 / tf.norm(outputs.values)
                norm_low_state_gnn = tf.sparse.map_values(tf.multiply, outputs, normaliz_gnn)
                overlap_temp = tf.norm(calculate_sparse_overlap(lowest_eigenstate_as_sparse, norm_low_state_gnn))
                overlap_in_time.append(overlap_temp.numpy())
                del outputs, norm_low_state_gnn, overlap_temp

        endtime = time.time() - start_time
        print("[TRACEMALLOC] Stopped tracing in outer_training_mc.")

        return endtime, energies, loss_vectors, overlap_in_time, magnetizations

# --- Data Classes ---
@dataclass(frozen=True)
class GraphParams:
    graphType: str="2dsquare"
    n:int =2
    m: int=2
    sublattice: str = "Neel"

@dataclass(frozen=True)
class SimParams:
    beta: float = 0.07
    batch_size: int =4
    learning_rate: float= 7e-5
    outer_loop:int=10
    inner_loop:int=3

@dataclass
class Hyperams:
    symulation_type: str="VMC"
    graph_params: GraphParams=field(default_factory=GraphParams)
    sim_params: SimParams = field(default_factory=SimParams)
    ansatz: str = "GNN2simple"
    ansatz_params: dict = field(default_factory=lambda: {"hidden_size": 128, "output_emb_size": 64})

# --- Minimal VMC Run ---
def minimal_vmc_run():
    """Runs a minimal example of the outer_training_mc function."""
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

    model_w = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params)
    model_fix = initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params)

    graph_tuples_var = initialize_graph_tuples(hyperparams.sim_params.batch_size, graph, subl)
    graph_tuples_fix = initialize_graph_tuples(hyperparams.sim_params.batch_size, graph, subl)

    endtime, energies, loss_vectors, overlap_in_time, magnetizations = outer_training_mc(
        outer_steps=hyperparams.sim_params.outer_loop,
        inner_steps=hyperparams.sim_params.inner_loop,
        graph=graph,
        beta=hyperparams.sim_params.beta,
        initial_learning_rate=hyperparams.sim_params.learning_rate,
        model_w=model_w,
        model_fix=model_fix,
        graph_tuples_var=graph_tuples_var,
        graph_tuples_te=graph_tuples_fix,
        lowest_eigenstate_as_sparse=lowest_eigenstate_as_sparse
    )

    print("\n========= Simulation Summary =========")
    print(f"Total simulation time: {endtime} seconds")
    print(f"Last energy in time: {energies[-1] if energies else None}")
    print(f"Last overlap in time: {overlap_in_time[-1] if overlap_in_time else None}")
    print(f"Last magnetization in time: {magnetizations[-1] if magnetizations else None}")
    print("======================================\n")

# --- Main Execution ---
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.run_functions_eagerly(True)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")

    tf.debugging.enable_check_numerics()
    minimal_vmc_run()