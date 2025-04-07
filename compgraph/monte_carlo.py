import multiprocessing as mp
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import time
import sonnet as snt
from compgraph.cg_repr import config_hamiltonian_product, graph_tuple_to_config_hamiltonian_product_update
from compgraph.useful import generate_graph_tuples_configs, graph_tuple_toconfig, update_graph_tuple_config, generate_graph_tuples_configs_tf
from compgraph.tensorflow_version.hamiltonian_operations import graph_hamiltonian_jit,graph_hamiltonian_jit_xla
from compgraph.tensorflow_version.graph_tuple_manipulation import get_single_graph_from_batch  

from typing import Tuple
from graph_nets.graphs import GraphsTuple


# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)
#Todo make the 
from memory_profiler import profile as mprofile

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
    def __init__(self, model, current_tuple, template=None,beta=None, graph=None, initialized=False, edge_pairs=None):
        self.model = model
        self.tuple = current_tuple
        self.beta = beta
        self.graph = graph
        self.edge_pairs = edge_pairs
        self.initialized=initialized
        self.template=template

        if not self.initialized:
            # print(current_tuple, self.tuple)
            self.model(current_tuple)
            self.initialized=True 
    def update_model(self, model):
        self.model = model
    # @tf.function(jit_compile=True)
    def evaluate_model(self, graph_tuple):
        output=self.model(graph_tuple)
        amplitude, phase = output[0][0],output[0][1]
        return tf.complex(real=amplitude * tf.cos(phase),
                        imag=amplitude * tf.sin(phase))

    def time_evoluted_config_amplitude(self, graph_tuple):
        # Obtain the nonzero graph tuples and their amplitudes.
        graph_tuples_nonzero, amplitudes_gt = graph_tuple_to_config_hamiltonian_product_update(graph_tuple, self.graph)
        
        # Print the number of nonzero graph tuples using Python's len()
        # tf.print("Number of nonzero graph tuples:", len(graph_tuples_nonzero))
        
        final_amplitude = []
        for i, gt in enumerate(graph_tuples_nonzero):
            # Debug: print the nodes of the current graph tuple.
            # tf.print("Graph tuple", i, "nodes:", gt.nodes)
            
            # Evaluate the model on the current graph tuple.
            output = self.model(gt)
            # tf.print("Model output for tuple", i, ":", output)
            
            amplitude, phase = output[0][0], output[0][1]
            # tf.print("Amplitude:", amplitude, "Phase:", phase)
            
            amplitude = amplitude * amplitudes_gt[i]
            complex_coefficient = tf.complex(real=amplitude * tf.cos(phase),
                                            imag=amplitude * tf.sin(phase))
            # tf.print("Complex coefficient for tuple", i, ":", complex_coefficient)
            final_amplitude.append(complex_coefficient)
        
        beta = -1.0 * self.beta
        total_amplitude = tf.multiply(beta, tf.reduce_sum(tf.stack(final_amplitude)))
        
        # Evaluate the baseline output.
        complex_coefficient = self.evaluate_model(graph_tuple)
        # tf.print("Baseline model output:", complex_coefficient)
        
        total_amplitude = tf.add(complex_coefficient, total_amplitude)
        # tf.print("Total amplitude:", total_amplitude)
        
        return total_amplitude
    
    @tf.function()
    def ite_step(self, new_graphs, all_amplitudes, initial_gt):
        baseline_output = self.evaluate_model(initial_gt)  # Ensure this takes a single graph

        # Evaluate the model on all new graphs in a single batch operation
        model_outputs =self.model(new_graphs)  # shape: [batch_size, 2]
        # Extract amplitude and phase from model output
        # return model_outputs

        amplitudes, phases = model_outputs[:, 0], model_outputs[:, 1]

        # return amplitudes, phases, all_amplitudes
        # Compute the complex coefficients
        complex_coefficients = tf.complex(
            real=amplitudes * tf.cos(phases),
            imag=amplitudes * tf.sin(phases)
        )
        beta = -1.0 * self.beta
        # Multiply by the Hamiltonian amplitudes
        weighted_coefficients = complex_coefficients * all_amplitudes
        # Scale by β and sum the contributions
        
        total_amplitude = beta * tf.reduce_sum(weighted_coefficients)    
        # Compute the baseline model output for the original graph

        # Add the baseline contribution
        total_amplitude = baseline_output + total_amplitude
        return total_amplitude
    
    @tf.function()
    def time_evoluted_config_amplitude_tf(self, graph_tuple, j2: float = 0.0):
        """Compute the time-evolved configuration and amplitude for a given configuration."""
        # Compute the new configurations and amplitudes
        # Compute all new configurations and their amplitudes in one call
        # tf.print("shape of edge pairs",tf.shape(self.edge_pairs))
        new_graphs, all_amplitudes = graph_hamiltonian_jit_xla(graph_tuple,self.edge_pairs, j2, self.template)
        # print("New Graphs generated by time evoluted config amplitudes", new_graphs)
        all_amplitudes=tf.cast(all_amplitudes, tf.complex64)
        # return all_amplitudes
        return self.ite_step(new_graphs, all_amplitudes, graph_tuple)

    def monte_carlo_update(self, N_sweeps, graph_tuple, approach):
        state = graph_tuple
        if approach == 'var':
            psi = self.evaluate_model(state)
        else:
            psi = self.time_evoluted_config_amplitude(state)
        for sweep_idx in range(N_sweeps):
            proposed_graph_tuple = propose_graph_tuple(state)
            if approach == 'var':
                psi_new = self.evaluate_model(proposed_graph_tuple)
            else:
                psi_new = self.time_evoluted_config_amplitude(proposed_graph_tuple)
            print("What is the step",sweep_idx, "the type is",tf.square(tf.abs(psi_new / psi)))
            p_accept = tf.minimum(tf.constant(1.0, dtype=tf.float64), tf.square(tf.abs(psi_new / psi)))
            state, psi = tf.cond(
                tf.less(tf.random.uniform([], dtype=tf.float64), p_accept),
                lambda: (proposed_graph_tuple, psi_new),
                lambda: (state, psi)
            )
        return state, psi    
    @staticmethod
    @tf.function(jit_compile=True)
    def propose_graph_batch(graphs_batch: GraphsTuple) -> GraphsTuple:
        """Propose new configurations for entire batch by flipping 1 node/graph"""
        # Get batch dimensions
        num_graphs = tf.shape(graphs_batch.n_node)[0]
        nodes_per_graph = graphs_batch.n_node[0]
        total_nodes = num_graphs * nodes_per_graph

        # Generate random flip indices [num_graphs]
        flip_indices = tf.random.uniform(
            shape=[num_graphs],
            maxval=nodes_per_graph,
            dtype=tf.int32
        )

        # Create global indices for scatter update
        graph_offsets = tf.range(num_graphs) * nodes_per_graph
        global_indices = graph_offsets + flip_indices  # [num_graphs]
        
        # Create updates tensor
        original_spins = tf.gather(graphs_batch.nodes[:, 0], global_indices)
        updates = -original_spins  # Flip spins
        
        # Apply scatter update
        update_indices = tf.stack([global_indices, tf.zeros_like(global_indices)], axis=1)
        new_nodes = tf.tensor_scatter_nd_update(
            tensor=graphs_batch.nodes,
            indices=update_indices,
            updates=updates
        )

        return graphs_batch.replace(nodes=new_nodes)
    
    @staticmethod    
    @tf.function(jit_compile=True)
    def propose_graph_batch_exchange(graphs_batch):
        """Propose new configurations for entire batch by exchanging pairs of opposite spins.
        
        This preserves the total magnetization by swapping pairs of opposite spins rather than
        flipping individual spins. If no suitable exchange pair is found, the graph remains unchanged.
        """
        # Get batch dimensions
        num_graphs = tf.shape(graphs_batch.n_node)[0]
        nodes_per_graph = graphs_batch.n_node[0]
        
        # Create offsets for each graph in the batch
        graph_offsets = tf.range(num_graphs) * nodes_per_graph
        
        # For each graph, select a random node for potential exchange
        flip_indices1 = tf.random.uniform(
            shape=[num_graphs], maxval=nodes_per_graph, dtype=tf.int32
        )
        global_indices1 = graph_offsets + flip_indices1
        
        # Get the spins of the first selected nodes
        spins1 = tf.gather(graphs_batch.nodes[:, 0], global_indices1)
        
        # Select a second random node for each graph 
        flip_indices2 = tf.random.uniform(
            shape=[num_graphs], maxval=nodes_per_graph, dtype=tf.int32
        )
        global_indices2 = graph_offsets + flip_indices2
        
        # Get the spins of the second selected nodes
        spins2 = tf.gather(graphs_batch.nodes[:, 0], global_indices2)
        
        # Create a mask for valid exchanges (nodes have opposite spins and are different nodes)
        valid_exchange = tf.logical_and(
            tf.not_equal(global_indices1, global_indices2),  # Different nodes
            tf.equal(spins1, -spins2)                        # Opposite spins
        )
        
        # Filter indices to only include valid exchanges
        # We need to handle case where no valid exchange is found in a graph
        filtered_indices1 = tf.boolean_mask(global_indices1, valid_exchange)
        filtered_indices2 = tf.boolean_mask(global_indices2, valid_exchange)
        filtered_spins1 = tf.boolean_mask(spins1, valid_exchange)
        filtered_spins2 = tf.boolean_mask(spins2, valid_exchange)
        
        # Prepare for scatter updates
        update_indices1 = tf.stack([filtered_indices1, tf.zeros_like(filtered_indices1)], axis=1)
        update_indices2 = tf.stack([filtered_indices2, tf.zeros_like(filtered_indices2)], axis=1)
        
        # First swap: nodes1 gets values of nodes2
        new_nodes = tf.tensor_scatter_nd_update(
            tensor=graphs_batch.nodes,
            indices=update_indices1,
            updates=filtered_spins2
        )
        
        # Second swap: nodes2 gets values of nodes1
        new_nodes = tf.tensor_scatter_nd_update(
            tensor=new_nodes,
            indices=update_indices2,
            updates=filtered_spins1
        )
        
        return graphs_batch.replace(nodes=new_nodes)

    @staticmethod
    @tf.function(jit_compile=True)
    def calculate_acceptance_prob(psi_old: tf.Tensor, psi_new: tf.Tensor) -> tf.Tensor:
        """Vectorized acceptance probability calculation"""
        ratios = tf.abs(psi_new[:, 0] / psi_old[:, 0])  # Amplitude ratios
        return tf.minimum(1.0, tf.square(ratios))

    @staticmethod
    @tf.function(jit_compile=True)
    def update_batch_with_mask(current_batch, proposed_batch, accepted_mask):
        """
        For each graph in the batch, if accepted_mask[b] is True, take proposed_batch.nodes; else keep current_batch.nodes.
        Returns an updated GraphsTuple (only nodes change).
        """
        B = tf.shape(current_batch.n_node)[0]
        N = current_batch.n_node[0]
        D = tf.shape(current_batch.nodes)[1]
        
        # Reshape nodes fields: [B, N, D]
        current_nodes = tf.reshape(current_batch.nodes, [B, N, D])
        proposed_nodes = tf.reshape(proposed_batch.nodes, [B, N, D])
        
        # Expand accepted_mask for broadcasting: [B, 1, 1]
        accepted_mask_exp = tf.reshape(accepted_mask, [B, 1, 1])
        
        # Select proposed nodes where accepted, current nodes otherwise
        new_nodes = tf.where(accepted_mask_exp, proposed_nodes, current_nodes)
        new_nodes_flat = tf.reshape(new_nodes, [B * N, D])
        
        return current_batch.replace(nodes=new_nodes_flat)
    # Very IMPORTANT: 
    """
    The jit compilation is not working for the monte_carlo_update_on_batch function. If inserting jit the full loop is jit compiled
    Further this is retraced every time the function is called causing memory growth.
    Since the inference on monte carlo update on batch can be very quick and jit compilation is slow we should avoid to use jit compile 
    on this loop. It can not be done efficiently with my current knowledge 
    use mcmc_loop to generate plots of the problem
    """
    @tf.function()  # Add JIT here
    def monte_carlo_update_on_batch(self, GT_batch: GraphsTuple, N_sweeps: int) -> Tuple[GraphsTuple, tf.Tensor]:
        """Vectorized Monte Carlo update for entire batch"""
        # Initial evaluation we only care about the amplitudes, 
        # here we assume the model returns the amplitudes and the phases for each graph in the batch
        psi = self.model(GT_batch)
        current_psi = psi
        del psi
        current_batch=GT_batch
        shape=tf.shape(GT_batch.n_node)[0]
        print("This is the shape, dummy check for retracing!",shape)
        for _ in range(N_sweeps):
            # Propose new batch
            proposed_batch = self.propose_graph_batch(current_batch)
            psi_new = self.model(proposed_batch)
            # tf.print("What is the step",_)

            # Calculate acceptance probabilities
            p_accept = self.calculate_acceptance_prob(current_psi, psi_new)
            
            # Create acceptance mask
            rand_vals = tf.random.uniform(
                shape=[shape], 
                dtype=tf.float32
            )
            accept_mask = rand_vals < p_accept
            
            current_batch = self.update_batch_with_mask(current_batch,proposed_batch,accept_mask)
            # Update current psi values
            current_psi = tf.where(
                accept_mask[:, tf.newaxis], 
                psi_new, 
                current_psi
            )
            del proposed_batch, psi_new

        return current_batch, current_psi
    @tf.function()
    def monte_carlo_update_on_batchv2(self, GT_batch: GraphsTuple, N_sweeps: int) -> Tuple[GraphsTuple, tf.Tensor]:
        """Vectorized Monte Carlo update for entire batch with option for spin flip or exchange
        
        Args:
            GT_batch: Input batch of graphs
            N_sweeps: Number of Monte Carlo sweeps
            use_exchange: If True, use spin exchange; if False, use spin flip
        
        Returns:
            Updated batch and corresponding wave function amplitudes
        """
        # Initial evaluation
        psi = self.model(GT_batch)
        current_psi = psi
        del psi
        current_batch = GT_batch
        shape = tf.shape(GT_batch.n_node)[0]
        # print("This is the shape, dummy check for retracing!", shape)
        
        for _ in range(N_sweeps):
            proposed_batch = self.propose_graph_batch_exchange(current_batch)
            
            
            # Compute new wave function
            psi_new = self.model(proposed_batch)
            
            # Calculate acceptance probabilities
            p_accept = self.calculate_acceptance_prob(current_psi, psi_new)
            
            # Create acceptance mask
            rand_vals = tf.random.uniform(
                shape=[shape], 
                dtype=tf.float32
            )
            accept_mask = rand_vals < p_accept
            
            # Update current batch and psi values
            current_batch = self.update_batch_with_mask(current_batch, proposed_batch, accept_mask)
            current_psi = tf.where(
                accept_mask[:, tf.newaxis], 
                psi_new, 
                current_psi
            )
            del psi_new, proposed_batch

        return current_batch, current_psi
@tf.function()
def compute_phi_terms(batched_graphs: GraphsTuple, sampler: MCMCSampler):
    batch_size = tf.shape(batched_graphs.n_node)[0]
    # print("This is the batch size",batch_size)
    # print("This is the batched graphs",batched_graphs)
    # Map over each graph in the batch.

    def single_graph_phi(i):
        single_graph = get_single_graph_from_batch(batched_graphs, i)
        # print("This is the single graph",single_graph)
        return sampler.time_evoluted_config_amplitude_tf(single_graph, sampler.edge_pairs)
    
    # tf.map_fn will execute single_graph_phi for each index in [0, batch_size)
    return tf.map_fn(single_graph_phi, tf.range(batch_size), dtype=tf.complex64)

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
