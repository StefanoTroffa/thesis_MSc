import os, glob, re, warnings
import tensorflow as tf
import pandas as pd
import sonnet as snt
import numpy as np
from compgraph.tensorflow_version.model_loading import load_model_from_path,extract_hyperparams_from_path
from simulation.initializer import create_graph_from_ham, initialize_NQS_model_fromhyperparams
from compgraph.tensorflow_version.graph_tuple_manipulation import initialize_graph_tuples_tf_opt
from tensorboard.backend.event_processing import event_accumulator

def extract_run_data_from_tf(path, base_dir=None, tags=("training/energy_real", "training/std_energy", "training/staggered_magnetization_abs")):
    """
    Load the latest TensorBoard event file in the given path and extract time-series data for each tag.
    Returns a dict suitable for aggregation into a DataFrame, with series as lists.
    """
    # A) Pick latest event file
    event_files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {path}")

    latest_file = max(event_files, key=os.path.getmtime)

    # B) Extract run name (relative to some root)
    run_name = os.path.relpath(path, base_dir) if base_dir else path

    # C) Load tensor events
    ea = event_accumulator.EventAccumulator(latest_file, size_guidance={"tensors": 0})
    ea.Reload()

    available = ea.Tags().get("tensors", [])
    record = {"run_name": run_name}

    for tag in tags:
        if tag not in available:
            print(f"Warning: Tag '{tag}' not found in {latest_file}")
            record[tag] = []
            continue

        steps = []
        values = []

        for evt in ea.Tensors(tag):
            try:
                val = tf.make_ndarray(evt.tensor_proto).item()
            except Exception as e:
                print(f"Error decoding {tag} @ step {evt.step}: {e}")
                continue
            steps.append(evt.step)
            values.append(val)

        record[tag] = values
        if "steps" not in record:
            record["steps"] = steps

    return record


def batch_extract_tf_events(paths, tags=("training/energy_real", "training/std_energy", "training/staggered_magnetization_abs")):
    """
    Apply extract_run_data to a list of run paths and return a combined DataFrame.
    """
    base_dir = os.path.commonpath(paths)
    results = []

    for p in paths:
        try:
            row = extract_run_data_from_tf(p, base_dir=base_dir, tags=tags)
            results.append(row)
        except Exception as e:
            print(f"Skipping {p}: {e}")

    return pd.DataFrame(results)

def directory_filter(pattern=r'03_03', path=None):
    """
    Filter function to check if a directory contains a specific pattern.
    """
    return re.search(pattern, path) is not None

def get_filtered_directories(base_dir="saving_logs/**/GNN*", pattern=r'03_03'):
    """
    Get a list of directories that match the given pattern.
    """
    run_dirs = glob.glob(base_dir, recursive=True)
    filtered_dirs = [d for d in run_dirs if os.path.isdir(d) and directory_filter(pattern, path=d)]
    return filtered_dirs


def safe_decode(x):
    try:
        return x.decode() if isinstance(x, bytes) else str(x)
    except: 
        return str(x) # Fallback

def extract_specific_hyperparams(config_str):
    hyperparams = {}
    # Regex to find specific parameters. Adjust patterns based on your config string format.
    beta_match = re.search(r'beta:\s*(\d+\.?\d*)', config_str)
    lr_match = re.search(r'learning_rate:\s*(\d+\.?\d*[eE]?-?\d*)', config_str)
    gnn_match = re.search(r'(GNN\d+adv|GNNprocnorm)_h\d+_e\d+_K\d+', config_str) # Example GNN architecture match
    seed_match = re.search(r'seed:\s*(\d+)', config_str)

    hyperparams['beta'] = float(beta_match.group(1)) if beta_match else np.nan
    hyperparams['lr'] = float(lr_match.group(1)) if lr_match else np.nan
    hyperparams['seed'] = int(seed_match.group(1)) if seed_match else np.nan
    return hyperparams

def filter_valid(df, column="training/energy_real", n_steps_min=1000,treshold=0.01):
    """
    Filter DataFrame rows based on a threshold for the final energy.
    """
    df_filt=df[df[column].apply(len) >= n_steps_min]
    # Add final values
    excluded_columns = ['steps', 'run_name', 'configuration/hyperparameters']
    for col in df_filt.columns:
        if col not in excluded_columns:
            print(col)
            col_name= "final_"+col.split("/")[-1]
            print(col_name)
            df_filt[col_name]= df_filt[col].apply(lambda x: x[-1] if len(x)>0 else pd.NA)
    
    df_filt=df_filt[df_filt["final_energy_real"]<treshold]
    df_runs_seeded_copy = df_filt.copy()
    df_runs_seeded_copy['config_str'] = df_runs_seeded_copy['configuration/hyperparameters'].apply(safe_decode)

    hyperparams_df = df_runs_seeded_copy['config_str'].apply(lambda s: pd.Series(extract_specific_hyperparams(s)))
    df_runs_seeded_copy = pd.concat([df_runs_seeded_copy, hyperparams_df], axis=1)

    return df_runs_seeded_copy
    
def load_models_for_analysis(df_sim_filt,filtered_dirs):
    warnings.filterwarnings("ignore", category=FutureWarning)
    tf.get_logger().setLevel('ERROR')
    models = {}

    df_sim_filt_seeded = df_sim_filt[~df_sim_filt['seed'].isna()]
    dirs_for_checkpoints = [
        d for d in filtered_dirs
        if any(run in d for run in df_sim_filt_seeded['run_name'].unique().tolist())
    ]
    for idx, d in enumerate(dirs_for_checkpoints):
        ckpt_paths=sorted(glob.glob(d+'/checkpoints/*.index', recursive=True))
        ckpt_idxs = sorted(glob.glob(os.path.join(d, 'checkpoints', '*.index')))
        ckpt_path = ckpt_idxs[-2][:-6]   # rimuovo '.index'
        print("→ checkpoint:", ckpt_path)
        hyperparams=extract_hyperparams_from_path(ckpt_path)
        model_temp=initialize_NQS_model_fromhyperparams(hyperparams.ansatz, hyperparams.ansatz_params,0)
        optimizer_temp=snt.optimizers.Adam(hyperparams.sim_params.learning_rate,0.9,0.99)
        n_hyp,m_hyp=hyperparams.graph_params.n,hyperparams.graph_params.m
        graph,subl=create_graph_from_ham(hyperparams.graph_params.graphType,(n_hyp,m_hyp)
                                        ,sublattice=hyperparams.graph_params.sublattice)

        GT_Batch_init=initialize_graph_tuples_tf_opt(128, graph, sublattice_encoding=subl)    
        # prova forward prima del caricamento
        print()
        _ = model_temp(GT_Batch_init)
        print(f"✘ Model initialized, psi_val[0]={_[0]}")
        # carica pesi
        load_model_from_path(
            model=model_temp,
            checkpoint_path=ckpt_path,  # load_model cerca i file .index ecc qui
            optimizer=optimizer_temp
        )

        # verifica forward dopo
        psi_val = model_temp(GT_Batch_init)[0]
        print(f"✔ Model loaded from {d},\n, psi_val[0]={psi_val}")

        # chiave unica: basename + beta + bs
        run_key = os.path.basename(d) +f"_b{hyperparams.sim_params.beta}_bs{hyperparams.sim_params.batch_size}_lr{hyperparams.sim_params.learning_rate}"+"_"+str(idx)
        models[run_key] = {
            "model": model_temp,
            "hyperparams": hyperparams
        }

    print("\n Loaded models:", list(models.keys()))
    return models