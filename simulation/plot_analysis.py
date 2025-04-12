"""Print a summary of a data file. It can be a summary_*.pkl or a data_*.pkl.gz file """
import pickle
import pandas as pd
import numpy as np
import os, sys
import argparse as args

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
def extract_configs_as_indices(GT_batch):
    """
    Extract configurations from GraphTuple and convert to integer indices
    for comparison with the full Hilbert space.
    """
    # Extract spin configurations (assuming they're in the first column of nodes)
    configs = tf.reshape(GT_batch.nodes[:, 0], (-1, GT_batch.n_node[0]))
    
    # Convert from [-1,1] format to [0,1] format
    configs_01 = tf.cast((configs + 1) / 2, tf.int32)
    
    # Convert each configuration to an integer index in the Hilbert space
    # Using binary representation: each config maps to a unique integer
    indices = tf.zeros(tf.shape(configs)[0], dtype=tf.int32)
    
    for i in range(tf.shape(configs)[1]):
        bit_value = tf.cast(configs_01[:, i], tf.int32) * (2 ** i)
        indices += bit_value
    
    return indices
def get_prob_amplitudes_from_mc_sampling(GT_batch_mc,num_nodes, model_w, sampler_var, GT_batch_complete, n_samples=500):
    """
    Extract the probability amplitudes from the Monte Carlo sampling.
    """
    num_bins = 2 ** num_nodes
    # Create a histogram (count per configuration index)
    hist_counts = np.zeros(num_bins)

    for i in range(n_samples):
        for l in range(num_nodes):
            GT_batch_mc, psi_coeff=sampler_var.monte_carlo_update_on_batch(model_w,GT_batch_mc)

        config_indices_np=extract_configs_as_indices(GT_batch_mc).numpy()
        for idx in config_indices_np:
            hist_counts[idx] += 1
    if not GT_batch_complete is None:
        psi_model=model_w(GT_batch_complete)
        prob_ampl=tf.abs(tf.complex(psi_model[:,0] * tf.cos(psi_model[:,1]),psi_model[:,0] * tf.sin(psi_model[:,1])))**2
        prob_coeff=prob_ampl/tf.reduce_sum(prob_ampl)
    return hist_counts, prob_coeff


def plot_data(df, column, title, xlabel, ylabel):
    plt.figure()
    plt.plot(df[column])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def main(args):
    if os.path.isfile(args.fname):
        fname_base = os.path.basename(args.fname)
        name, ext = os.path.splitext(fname_base)
        
        if ext == ".pkl":
            # Load DataFrame from pickle file
            data = pd.read_pickle(args.fname)
            print("DataFrame loaded successfully.")
            
            # Assuming data is a DataFrame and has columns named 'energies', 'loss_vectors', and 'overlap_in_time'
            if isinstance(data, pd.DataFrame):
                plot_data(data, 'energies', 'Energy Evolution', 'Training Step', 'Energy')
                plot_data(data, 'loss_vectors', 'Loss Evolution', 'Training Step', 'Loss')
                plot_data(data, 'overlap_in_time', 'Overlap Evolution', 'Training Step', 'Overlap')
            else:
                print("Error: Data loaded is not in expected DataFrame format.")
        
        else:
            print(f"Unsupported file format: {ext}", file=sys.stderr)
    else:
        print(f"File '{args.fname}' does not exist", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data from a pickle file and plot.")
    parser.add_argument("fname", type=str, help="Filename of the pickle file to load")
    args = parser.parse_args()
    main(args)
