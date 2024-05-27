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
