import os, glob
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.compat.proto import tensor_pb2
import tensorflow as tf
import pandas as pd


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
