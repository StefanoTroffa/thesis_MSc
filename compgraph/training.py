import time
import numpy as np
import tensorflow as tf
import quimb as qu
import sonnet as snt

from compgraph.tensor_wave_functions import montecarlo_logloss_overlap_time_evoluted, quimb_vec_to_sparse, variational_wave_function_on_batch, sparse_tensor_exp_energy, calculate_sparse_overlap,time_evoluted_wave_function_on_batch
from memory_profiler import profile as mprofile
# from compgraph.monte_carlo import MCMCSampler
import line_profiler
import tracemalloc

from compgraph.useful import graph_tuple_list_to_configs_list, copy_to_non_trainable, compare_sonnet_modules, sites_to_sparse_updated, create_amplitude_frequencies_from_graph_tuples,sparse_list_to_configs
from compgraph.monte_carlo import MCMCSampler, stochastic_gradients, stochastic_energy
from simulation.initializer import create_graph_from_ham, format_hyperparams_to_string, initialize_NQS_model_fromhyperparams, initialize_graph_tuples, initialize_hamiltonian_and_groundstate
import multiprocessing as mp
#from joblib import Parallel, delayed
from multiprocessing import Pool

from compgraph.cg_repr import config_hamiltonian_product, graph_tuple_to_config_hamiltonian_product_update
# import line_profiler
from compgraph.useful import graph_tuple_toconfig, generate_graph_tuples_configs_tf
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)
#Todo make the 
def propose_graph_tuple(graph_tuple):
    """
    Propose a new graph tuple by flipping the spin of a randomly selected node.
    Args: graph_tuple (tf.Tensor): The input graph tuple.
    Returns: tf.Tensor: The proposed graph tuple with the spin of one node flipped.
    """
    proposed_nodes = graph_tuple.nodes.numpy().copy()  # Convert to numpy array for mutability
    i = np.random.randint(len(proposed_nodes))  # Choose a random node
    proposed_nodes[i, 0] *= -1  # Flip the spin at this node
    return graph_tuple.replace(nodes=tf.constant(proposed_nodes))
class MCMCSampler:
    def __init__(self, model, current_tuple, beta=None, graph=None, initialized=False):
        self.model = model
        self.tuple = current_tuple
        self.beta = beta
        self.graph = graph
        # self.sublattice_encoding = sublattice_encoding
        self.initialized=initialized
        if not self.initialized:
            self.model(current_tuple)
            self.initialized=True 
    def update_model(self, model):
        self.model = model
    def evaluate_model(self, graph_tuple):
        amplitude, phase = self.model(graph_tuple)[0]
        return tf.complex(real=amplitude* tf.cos(phase), imag=amplitude * tf.sin(phase))
    def time_evoluted_config_amplitude(self, graph_tuple):
        graph_tuples_nonzero, amplitudes_gt = graph_tuple_to_config_hamiltonian_product_update(graph_tuple, self.graph)
        final_amplitude = []
        for i, gt in enumerate(graph_tuples_nonzero):
            amplitude, phase = self.model(gt)[0]
            amplitude *= amplitudes_gt[i]
            complex_coefficient = tf.complex(real=amplitude* tf.cos(phase), imag=amplitude * tf.sin(phase))
            final_amplitude.append(complex_coefficient)
        beta = -1. * self.beta
        total_amplitude = tf.multiply(beta, tf.reduce_sum(tf.stack(final_amplitude)))
        complex_coefficient = self.evaluate_model(graph_tuple)
        total_amplitude = tf.add(complex_coefficient, total_amplitude)
        return total_amplitude

    # @profile
    # @tf.function
    def monte_carlo_update(self, N_sweeps, graph_tuple, approach):
        self.tuple = graph_tuple
        state=self.tuple
        if approach == 'var':
            psi = self.evaluate_model(state)

            for _ in tf.range(N_sweeps):
                proposed_graph_tuple = propose_graph_tuple(state)
                psi_new = self.evaluate_model(proposed_graph_tuple)
                p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.abs(psi_new / psi)**2)
                if tf.random.uniform([], dtype=tf.float64) < p_accept:
                    psi=psi_new

                    state = proposed_graph_tuple
        elif approach == 'te':
            psi = self.time_evoluted_config_amplitude(state)

            for _ in tf.range(N_sweeps):
                proposed_graph_tuple = propose_graph_tuple(state)
                psi_new = self.time_evoluted_config_amplitude(proposed_graph_tuple)
                p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.abs(psi_new / psi)**2)

                if tf.random.uniform([], dtype=tf.float64) < p_accept:
                    psi=psi_new
                    state = proposed_graph_tuple

        self.tuple = state
        return state, psi


def stochastic_energy(model_var:MCMCSampler, graph, graph_tuple_configs, frequencies=None, J2=None):
    energy=0.+0.j
    local_energies=[]
    # If frequencies are not provided, assume uniform sampling (all frequencies set to 1/N)
    if frequencies is None:
        frequencies = tf.ones(len(graph_tuple_configs), dtype=tf.complex128)/len(graph_tuple_configs)
    if J2==None:
        J2=0.
    for idx, gt in enumerate(graph_tuple_configs):
        temp = 0. + 0.j
        # Evaluate the wavefunction for the current graph tuple
        psi_coeff = model_var.evaluate_model(gt)

        # Convert graph tuple to configuration
        config_gt = graph_tuple_toconfig(gt)

        # Get the non-zero Hamiltonian product configurations and their coefficients
        configurations_nonzero, coefficients = config_hamiltonian_product(config_gt, graph, J2)

        # Generate graph tuples for the non-zero configurations
        graph_tuples_nonzero = generate_graph_tuples_configs_tf(gt, configurations_nonzero)

        # Loop over the non-zero configurations to calculate their contributions to the energy
        for idx_nonzero, nonzero_gt in enumerate(graph_tuples_nonzero):
            # Evaluate the model's wavefunction for the non-zero graph tuple
            psi_coeff_nonzero = model_var.evaluate_model(nonzero_gt)

            # Add the contribution to the energy: coefficient * (psi(s') / psi(s))
            temp += coefficients[idx_nonzero] * (psi_coeff_nonzero / psi_coeff)

        # Accumulate energy with frequencies
        energy += temp * frequencies[idx]
        local_energies.append(temp)

    return energy, local_energies


def stochastic_overlap(sampler_var, sampler_te, coeff_te_on_te, unique_tuples_var, unique_tuples_te, frequencies_var, frequencies_te):
    """
    Compute the stochastic overlap between the wavefunctions of sampler_var and sampler_te.
    
    Args:
        sampler_var: MCMC sampler for the wavefunction psi.
        sampler_te: MCMC sampler for the wavefunction phi.
        graph_tuples_var: Graph tuples for sampler_var.
        graph_tuples_te: Graph tuples for sampler_te.
        frequencies_var: Frequencies for psi (sampler_var).
        frequencies_te: Frequencies for phi (sampler_te).
    
    Returns:
        The stochastic overlap between psi and phi.
    """
    overlap_psi_phi = 0.0 + 0.j
    overlap_phi_psi = 0.0 + 0.j
    norm_psi = 0.0 + 0.j
    norm_phi = 0.0 + 0.j
    
    # Iterate through graph tuples and accumulate overlap terms
    for idx, graph_tuple_var in enumerate(unique_tuples_var):
        psi_coeff = sampler_var.evaluate_model(graph_tuple_var)
        phi_coeff= sampler_te.time_evoluted_config_amplitude(graph_tuple_var)
        # Compute the ratio for psi -> phi and vice versa
        overlap_psi_phi += (phi_coeff / psi_coeff) * frequencies_var[idx]
        norm_psi += frequencies_var[idx]

    for idx, graph_tuple_te in enumerate(unique_tuples_te):
        psi_coeff=sampler_var.evaluate_model(graph_tuple_te)
        phi_coeff = coeff_te_on_te[idx]
        overlap_phi_psi += (psi_coeff / phi_coeff) * frequencies_te[idx]
        norm_phi += frequencies_te[idx]
        # Accumulate norms
        

    # Compute the final overlap using the formula
    overlap = (overlap_phi_psi*overlap_psi_phi) / (norm_psi * norm_phi)
    print(f"norms: phi{norm_phi}, psi:{norm_psi}")
    return overlap
def copy_and_perturb_weights(sampler_var, sampler_te, perturbation_scale=1e-4, show_changes=False):
    """
    Copy the parameters from sampler_var.model to sampler_te.model and perturb the weights
    to make sure they are different.
    
    Args:
        sampler_var: The original sampler whose model weights are copied.
        sampler_te: The target sampler to which the weights are copied and perturbed.
        perturbation_scale: The standard deviation of the normal noise added as a perturbation.

    Returns:
        sampler_te: The sampler with the perturbed weights.
    """
    # # Copy the weights
    # for var_param, te_param in zip(sampler_var.model.trainable_variables, sampler_te.model.trainable_variables):
    #     te_param.assign(var_param)

    # Apply a small perturbation to sampler_te's weights to ensure the models are different
    if show_changes:
            
        for param in sampler_var.model.trainable_variables[:1]:
            print('before var param',param)
            
    for param in sampler_te.model.variables:
        perturbation = tf.random.normal(param.shape, mean=0.0, stddev=perturbation_scale, dtype=param.dtype)  # Match dtype
        # print("before pert",param)
        param.assign_add(perturbation)
        # print('after',param)
    if show_changes:

        for param in sampler_var.model.trainable_variables[:1]:
            print('after',param)
                    
    return sampler_te
def stochastic_gradients(sampler_var,sampler_te, unique_tuples_var,freq_ampl_var):
    with tf.GradientTape() as tape:
        tape.watch(sampler_var.model.trainable_variables)
        psi_terms= tf.stack([sampler_var.evaluate_model(gt) for gt in unique_tuples_var])
        phi_terms= tf.stack([sampler_te.time_evoluted_config_amplitude(gt) for gt in unique_tuples_var])
        # print(psi_terms)

        log_psi=tf.math.log(tf.math.conj(psi_terms))
        ratio_phi_psi = tf.stop_gradient(phi_terms / psi_terms)
        # print(ratio_phi_psi, phi_terms)
        # log_psi_gradients=tape.gradient(log_psi, sampler_var.model.trainable_variables)
        stoch_estimation=freq_ampl_var*log_psi -ratio_phi_psi*log_psi*freq_ampl_var/tf.reduce_sum(ratio_phi_psi*freq_ampl_var)
        stoch_gradients=tape.gradient(tf.reduce_sum(stoch_estimation), sampler_var.model.trainable_variables)
        # print(stoch_gradients[0])
    return stoch_gradients    
# import line_profiler
# import atexit
# profile0 = line_profiler.LineProfiler()
# atexit.register(profile0.print_stats)
# @profile0
def inner_training(model_var, model_fix_for_te, graph_batch_var,graph_batch_te, optimizer, beta,sublattice, graph):

    with tf.GradientTape() as tape:
        output = variational_wave_function_on_batch(model_var, graph_batch_var)
        te_wave_on_te= time_evoluted_wave_function_on_batch(model_fix_for_te,beta,graph_batch_te,graph, sublattice)
        tape.watch(model_var.trainable_variables)
        #print("output: \n", output)
            
        loss = montecarlo_logloss_overlap_time_evoluted(te_wave_on_te, graph_batch_te, model_var, model_fix_for_te, graph_batch_var, beta, graph, sublattice)
        
    #print("Is it lossing: \n", loss, type(loss))
    gradients = tape.gradient(loss, model_var.trainable_variables)
    #print("are model variables and gradients two lists?", type(model.trainable_variables), type(gradients))

    #for var, grad in zip(model.trainable_variables, gradients):
    #    print(f"{var.name}: Gradient {'is None' if grad is None else 'exists'}")
    optimizer.apply(gradients, model_var.trainable_variables)
    
    return output, loss

# @mprofile
def outer_training(outer_steps, inner_steps, sublattice_encoding, graph,
                   lowest_eigenstate_as_sparse, beta, initial_learning_rate, model_w, model_fix, graph_tuples_var, graph_tuples_te):
    """
    Conducts the outer training loop for a variational quantum simulation, adjusting model weights
    to minimize energy and maximize overlap with target states obtained through discrete immaginary time evolution across multiple training cycles.

    Args:
        outer_steps (int): Number of outer loop steps to perform, each consisting of multiple inner training steps.
        inner_steps (int): Number of inner training steps within each outer step, focusing on refining model parameters.
        sublattice_encoding (numpy.array): Encodes the sublattice configuration, crucial for defining interactions within the lattice.
        graph (networkx.Graph): The graph representing the lattice structure of the system under study.
        lowest_eigenstate_as_sparse (tf.Tensor): The target state tensor, representing the system's expected lowest energy state in sparse format.
        beta (float): Learning rate modifier or inverse temperature parameter used in certain quantum simulation algorithms.
        initial_learning_rate (float): Initial learning rate for the optimizer.
        model_w: The variational model being optimized, writable and updated during training.
        model_fix: A fixed reference model used for certain calculations, not updated during training.
        graph_tuples_var (list): List of graph tuple configurations for the variational model.
        graph_tuples_te (list): List of graph tuple configurations for the fixed model.

    Returns:
        tuple: Contains the total elapsed time for the training, lists of energies, loss values, and overlap measurements through training.

    Description:
        The function initializes models and an optimizer, then enters a loop where it conducts training iterations,
        measures performance metrics, and adjusts model parameters. Key measurements include energy (how well the
        model's output aligns with physical expectations), loss (a measure of error in the model's output), and overlap
        (how closely the model's output state matches the desired quantum state).
    """

    start_time = time.time()  # Record start time for overall training duration tracking

    optimizer_snt = snt.optimizers.Adam(initial_learning_rate)  # Initialize the optimizer with the given learning rate

    # Lists to store metrics for analysis
    energies = []
    loss_vectors = []
    overlap_in_time = []

    # Initialize models with the first set of graph tuples to set dimensions and weights
    initialize_model_w = model_w(graph_tuples_var[0])
    initialize_model_fix = model_fix(graph_tuples_te[0])
    start=0
    
    # Outer loop for global training iterations
    for step in range(outer_steps):

        # Copy weights from writable to fixed model to maintain consistency
        copy_to_non_trainable(model_w, model_fix)

        # Inner loop for detailed optimization steps
        for innerstep in range(inner_steps):
            # Perform training step and compute metrics
            outputs, loss = inner_training(model_w, model_fix, graph_tuples_var, graph_tuples_te, optimizer_snt, beta, sublattice_encoding, graph)

            #if innerstep % 2 == 0:  # Conditional logging for visibility on progress
                #print(f"Step {step}.{innerstep}; Loss: {loss.numpy()}")

            # Normalization and energy calculation
            normaliz_gnn = 1 / tf.norm(outputs.values)
            norm_low_state_gnn = tf.sparse.map_values(tf.multiply, outputs, normaliz_gnn)
            current_energy = sparse_tensor_exp_energy(outputs, graph, 0)
            overlap_temp = tf.norm(calculate_sparse_overlap(lowest_eigenstate_as_sparse, norm_low_state_gnn))

            # Store results for later analysis
            overlap_in_time.append(overlap_temp.numpy())
            energies.append(current_energy)
            loss_vectors.append(loss.numpy())

    endtime = time.time() - start_time  # Calculate total training time

    # Return collected metrics and time
    return endtime, energies, loss_vectors, overlap_in_time



# @mprofile
def outer_training_mc(outer_steps, inner_steps, graph,
                     beta, initial_learning_rate, model_w, model_fix, graph_tuples_var, graph_tuples_te,lowest_eigenstate_as_sparse=None):
    n_sites=len(graph_tuples_var[0].nodes)
    # Start memory tracing
    # tracemalloc.start()

    N_sweeps=(n_sites)//2
    # N_sweeps=5
    start_time = time.time()
    energies = []
    loss_vectors = []
    overlap_in_time = []
    magnetizations=[]
    # SEED = 42
    # np.random.seed(SEED)
    # tf.random.set_seed(SEED)    
    #TODO Sonnet wants the model to be initialized..., tbh this can be moved to MCMC sampler 
    initialize_model_w = model_w(graph_tuples_var[0])
    initialize_model_fix = model_fix(graph_tuples_te[0])
    # Initialize samplers
    sampler_var = MCMCSampler(model_w, graph_tuples_var[0])
    sampler_te = MCMCSampler(model_fix, graph_tuples_te[0], beta, graph)
    n_sites=len(graph_tuples_var[0].nodes[:,0])
    optimizer = snt.optimizers.Adam(initial_learning_rate)
    if n_sites<17:
        fhs = np.array([[int(x) for x in format(i, f'0{n_sites}b')] for i in range(2**(n_sites))]) * 2 - 1
        fh_gt=generate_graph_tuples_configs_tf(graph_tuples_var[0],fhs)
    for step in range(outer_steps):
        are_identical = compare_sonnet_modules(sampler_var.model, sampler_te.model)
        # print("The models are identical before copying:", are_identical)
        copy_to_non_trainable(sampler_var.model, sampler_te.model)
        are_identical2 = compare_sonnet_modules(sampler_var.model, sampler_te.model)   
        # print("The models are identical after copying:", are_identical2)
        # copy_to_non_trainable(model_w, model_fix)

        for innerstep in range(inner_steps):
            # outputs, loss = inner_training(model_w, model_fix, graph_tuples_var, graph_tuples_te, optimizer_snt, beta, sublattice_encoding, graph)
            graph_tuples_var, coeff_var_on_var = zip(*[sampler_var.monte_carlo_update(N_sweeps, graph_tuple,'var') for graph_tuple in graph_tuples_var])
            # sampler_var.update_model(model_w)
            configs_var=sites_to_sparse_updated(graph_tuple_list_to_configs_list(graph_tuples_var))

            # print(f'configs var: {configs_var}')
            # Temporarily modified in asking two times the graph_tuples var
            wave_function_var_on_var, freq_var = create_amplitude_frequencies_from_graph_tuples(graph_tuples_var, coeff_var_on_var)
            freq_ampl_var = np.array(freq_var.values) / len(graph_tuples_var)
            unique_tuples_var = generate_graph_tuples_configs_tf(graph_tuples_var[0], sparse_list_to_configs(freq_var.indices[:, 0], n_sites))

            stoch_gradients=stochastic_gradients(sampler_var,sampler_te,unique_tuples_var,freq_ampl_var)
            #print("check of new inner functions", outputs==outputs2, loss==loss2)
            optimizer.apply(stoch_gradients, sampler_var.model.trainable_variables)            
            stoch_energy=stochastic_energy(sampler_var,graph,unique_tuples_var,freq_ampl_var)

            energies.append(stoch_energy[0].numpy())
            # print('stoch energy',stoch_energy[0].numpy(), 'freq', freq_ampl_var)
            # **Compute the magnetization**
            def graph_tuple_toconfig_tf(graph_tuple):
                config= graph_tuple.nodes[:, 0]
                return config

            # Step 1: Convert all unique graph tuples to configurations
            configs = [graph_tuple_toconfig_tf(sample) for sample in unique_tuples_var]
            # configs is a list of tensors with shape (num_sites,)

            # Step 2: Stack configurations into a tensor
            configs_tensor = tf.stack(configs)  # Shape: (num_configs, num_sites)

            # Step 3: Compute S_z(s) for all configurations
            # Sum over spins for each configuration
            Sz_s = tf.reduce_sum(configs_tensor, axis=1)  # Shape: (num_configs,)
            # print(Sz_s)
            # Step 4: Compute weighted average of S_z(s)
            freq_ampl_var_tensor = tf.convert_to_tensor(freq_ampl_var, dtype=Sz_s.dtype)  # Shape: (num_configs,)

            # Compute total_Sz using vectorized operations
            total_Sz = tf.reduce_sum(Sz_s * freq_ampl_var_tensor)

            # Compute total frequency (should be 1.0 if frequencies are normalized)
            total_frequency = tf.reduce_sum(freq_ampl_var_tensor)

            # Step 5: Compute average magnetization per site
            N = tf.cast(tf.shape(configs_tensor)[1], Sz_s.dtype)
            M_z = total_Sz / total_frequency
            m_z = M_z / N  # per-site magnetization

            magnetizations.append(m_z.numpy())


        # # tracemalloc.take_snapshot()
        # # At the end of the training, take a snapshot and analyze memory usage
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[Top 10 memory usage lines]")
        # for stat in top_stats[:10]:
        #     print(stat)
        
        # Stop memory tracing
        if n_sites<17:
            outputs = variational_wave_function_on_batch(sampler_var.model, fh_gt)
            normaliz_gnn = 1 / tf.norm(outputs.values)
            norm_low_state_gnn = tf.sparse.map_values(tf.multiply, outputs, normaliz_gnn)
            overlap_temp = tf.norm(calculate_sparse_overlap(lowest_eigenstate_as_sparse, norm_low_state_gnn))

        overlap_in_time.append(overlap_temp.numpy())
    endtime = time.time() - start_time
    # tracemalloc.stop()

    return endtime, energies, loss_vectors, overlap_in_time, magnetizations
