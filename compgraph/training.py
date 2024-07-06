import time
import numpy as np
import tensorflow as tf
import quimb as qu
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, time_evoluted_wave_function_on_batch
from compgraph.tensor_wave_functions import montecarlo_logloss_overlap_time_evoluted, sparse_tensor_exp_energy, calculate_sparse_overlap, quimb_vec_to_sparse
import sonnet as snt
from compgraph.useful import copy_to_non_trainable  # Importing custom functions and model class
from compgraph.monte_carlo import parallel_monte_carlo_update, sequential_monte_carlo_update

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

def rolling_window_batch(graph_tuples, start, batch_size):
    end = start + batch_size
    if end <= len(graph_tuples):
        return graph_tuples[start:end]
    else:
        return graph_tuples[start:] + graph_tuples[:end - len(graph_tuples)]



def outer_training(outer_steps, inner_steps, sublattice_encoding, graph, batch_size,
                   lowest_eigenstate_as_sparse, beta, initial_learning_rate, model_w, model_fix, graph_tuples_var, graph_tuples_te):
    """
    Conducts the outer training loop for a variational quantum simulation, adjusting model weights
    to minimize energy and maximize overlap with target states obtained through discrete immaginary time evolution across multiple training cycles.

    Args:
        outer_steps (int): Number of outer loop steps to perform, each consisting of multiple inner training steps.
        inner_steps (int): Number of inner training steps within each outer step, focusing on refining model parameters.
        sublattice_encoding (numpy.array): Encodes the sublattice configuration, crucial for defining interactions within the lattice.
        graph (networkx.Graph): The graph representing the lattice structure of the system under study.
        batch_size (int): Number of graph states processed in each training batch.
        lowest_eigenstate_as_sparse (tf.Tensor): The target state tensor, representing the system's expected lowest energy state in sparse format.
        beta (float): Learning rate modifier or inverse temperature parameter used in certain quantum simulation algorithms.
        initial_learning_rate (float): Initial learning rate for the optimizer.
        model_w (tf.keras.Model): The variational model being optimized, writable and updated during training.
        model_fix (tf.keras.Model): A fixed reference model used for certain calculations, not updated during training.
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

        # Prepare for the next batch of data
        start += batch_size

    endtime = time.time() - start_time  # Calculate total training time

    # Return collected metrics and time
    return endtime, energies, loss_vectors, overlap_in_time
def outer_training_mc(outer_steps, inner_steps, sublattice_encoding, graph, batch_size,
                   lowest_eigenstate_as_sparse, beta, initial_learning_rate, model_w, model_fix, graph_tuples_var, graph_tuples_te):
    start_time = time.time()
    optimizer_snt = tf.keras.optimizers.Adam(initial_learning_rate)

    energies = []
    loss_vectors = []
    overlap_in_time = []
    start = 0

    for step in range(outer_steps):
        copy_to_non_trainable(model_w, model_fix)

        graph_tuples_var_batch = rolling_window_batch(graph_tuples_var, start, batch_size)
        graph_tuples_te_batch = rolling_window_batch(graph_tuples_te, start, batch_size)
        
        # Monte Carlo update for the batches using multiprocessing
        graph_tuples_var_batch = parallel_monte_carlo_update(graph_tuples_var_batch, model_w, N_sweeps=1, approach='var')
        graph_tuples_te_batch = parallel_monte_carlo_update(graph_tuples_te_batch, model_fix, N_sweeps=1, approach='te', beta=beta, graph=graph, sublattice_encoding=sublattice_encoding)

        for innerstep in range(inner_steps):
            outputs, loss = inner_training(model_w, model_fix, graph_tuples_var_batch, graph_tuples_te_batch, optimizer_snt, beta, sublattice_encoding, graph)

            normaliz_gnn = 1 / tf.norm(outputs.values)
            norm_low_state_gnn = tf.sparse.map_values(tf.multiply, outputs, normaliz_gnn)
            current_energy = sparse_tensor_exp_energy(outputs, graph, 0)
            overlap_temp = tf.norm(calculate_sparse_overlap(lowest_eigenstate_as_sparse, norm_low_state_gnn))

            overlap_in_time.append(overlap_temp.numpy())
            energies.append(current_energy)
            loss_vectors.append(loss.numpy())

        start = (start + batch_size) % len(graph_tuples_var)

    endtime = time.time() - start_time
    return endtime, energies, loss_vectors, overlap_in_time
