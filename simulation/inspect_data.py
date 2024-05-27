"""Print a summary of a data file. It can be a summary_*.pkl or a data_*.pkl.gz file """
import pickle
import pandas as pd
import numpy as np
import os, sys


def main():
    if args.fname is not None and os.path.isfile(args.fname):
        fname_base=os.path.basename(args.fname)
        name,ext=os.path.splitext(fname_base)
        if ext == ".pkl": 
            # We are looking at a pandas dataframe
            df = pd.read_pickle(args.fname)
            print(df)
        elif ext== ".npy":
            # We are looking at some vector stored as .npy. Not used yet
            vec_import=np.load(args.fname)
            print("Length: ({})".format(len(vec_import)))
            print(vec_import)
        
        else:
            #We don't know what to do
            print("Invalid file '{}'".format(args.fname),file=sys.stderr)
    else:
        print("File '{}' does not exist".format(args.fname),file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("fname",type=str,help="filename to inspect")
    args=parser.parse_args()

    main()