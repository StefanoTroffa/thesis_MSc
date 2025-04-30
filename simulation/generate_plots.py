import os
import re
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def parse_sim_info(sim_info_str):
    """
    Parse simulation info string into hyperparameter dict.
    Expected format: beta{beta}__bs_{batch_size}lr{learning_rate}_loop{outer_loop}x{inner_loop}_{gradient}
    """
    pattern = r'beta(?P<beta>[\d\.]+)__bs_(?P<batch_size>\d+)lr(?P<learning_rate>[\d\.eE\-\+]+)_loop(?P<outer_loop>\d+)x(?P<inner_loop>\d+)_(?P<gradient>\w+)'
    m = re.match(pattern, sim_info_str)
    if not m:
        return {}
    d = m.groupdict()
    return {
        'beta': float(d['beta']),
        'batch_size': int(d['batch_size']),
        'learning_rate': float(d['learning_rate']),
        'outer_loop': int(d['outer_loop']),
        'inner_loop': int(d['inner_loop']),
        'gradient': d['gradient']
    }


def scan_experiments(base_dir, size):
    """
    Walk through base_dir, find TensorBoard log dirs (containing event files),
    and parse out hyperparameters and ansatz info.
    Returns a list of dicts with keys: beta, batch_size, learning_rate,
    outer_loop, inner_loop, gradient, ansatz, ansatz_params, log_dir
    """
    experiments = []
    for root, dirs, files in os.walk(base_dir):
        if any(f.startswith('events.out.tfevents') for f in files):
            parts = root.split(os.sep)
            try:
                idx = parts.index('system_Heisenberg')
                sim_info_type = parts[idx+2]
                ansatz_info = parts[idx+3]
            except (ValueError, IndexError):
                continue

            # split sim_info and simulation type
            tokens = sim_info_type.split('_')
            sim_type = tokens[-1]
            sim_info = '_'.join(tokens[:-1])
            hyper = parse_sim_info(sim_info)
            if not hyper:
                continue
            hyper['simulation_type'] = sim_type

            # parse ansatz info: {ansatz}_h{hidden}_e{emb}_K{K}
            ai = ansatz_info.split('_')
            if len(ai) != 4:
                continue
            hyper['ansatz'] = ai[0]
            hyper['ansatz_params'] = {
                'hidden_size': int(ai[1].lstrip('h')),
                'output_emb_size': int(ai[2].lstrip('e')),
                'K_layer': int(ai[3].lstrip('K'))
            }

            hyper['log_dir'] = root
            experiments.append(hyper)
    return experiments


def has_energy(log_dir):
    """
    Check if TensorBoard logs in log_dir contain an 'energy_real' scalar tag.
    """
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        print(f"Tags in {log_dir}: {tags}")
        return any('energy_real' in t for t in tags)
    except Exception:
        return False


def select_varying_parameter():
    """
    Prompt user to choose which hyperparameter to vary.
    """
    choices = ['beta', 'batch_size', 'learning_rate', 'inner_loop']
    for i, c in enumerate(choices, 1):
        print(f"{i}. {c}")
    idx = int(input("Select parameter to vary (1-4): ")) - 1
    return choices[idx]


def group_experiments(experiments, varying_param):
    """
    Group experiments by fixed hyperparameters (all except varying_param).
    Returns dict mapping fixed_params tuple -> list of experiments.
    """
    groups = defaultdict(list)
    key_params = ['beta', 'batch_size', 'learning_rate', 'outer_loop', 'inner_loop', 'gradient']
    for exp in experiments:
        fixed = tuple((k, exp[k]) for k in key_params if k != varying_param)
        groups[fixed].append(exp)
    return dict(groups)


def load_metrics(log_dir):
    """
    Load energy_real and std_energy scalars from TensorBoard logs.
    Returns (steps, energies, stds).
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # dynamic tag detection
    tags = ea.Tags().get('scalars', [])
    energy_tag = next(t for t in tags if 'energy_real' in t)
    std_tag = next(t for t in tags if 'std_energy' in t)

    e = ea.Scalars(energy_tag)
    std = ea.Scalars(std_tag)
    steps = [p.step for p in e]
    energy = [p.value for p in e]
    std_vals = [p.value for p in std]
    return steps, energy, std_vals


def plot_experiments(data_list, values, varying_param):
    """
    Plot energy vs step with error bars for each experiment.
    data_list: list of (steps, energy, std)
    values: list of varying_param values corresponding to each data set
    """
    for (steps, energy, std), val in zip(data_list, values):
        plt.errorbar(steps, energy, yerr=std, label=f"{varying_param}={val}")
    plt.legend()
    plt.xlabel('Training step')
    plt.ylabel('Energy')
    plt.title(f"Energy vs Step varying {varying_param}")
    plt.show()


def analyse_varying(base_dir, ansatz, ansatz_params,size):
    """
    High-level function to scan experiments, let user select a parameter to vary,
    choose a fixed-parameter group, and plot results.
    Only retains runs that have an energy_real tag.
    """
    # scan and filter by ansatz
    exps = scan_experiments(base_dir,size)
    exps = [e for e in exps if e['ansatz'] == ansatz and e['ansatz_params'] == ansatz_params]

    # filter out runs missing energy_real
    valid_exps = []
    for e in exps:
        if has_energy(e['log_dir']):
            valid_exps.append(e)
        else:
            print(f"Skipping {e['log_dir']} (no energy_real tag)")
    exps = valid_exps

    if not exps:
        print("No valid experiments found matching the given ansatz and parameters.")
        return

    # prompt and group
    varying = select_varying_parameter()
    groups = group_experiments(exps, varying)
    if not groups:
        print(f"No experiment groups found with varying {varying}.")
        return

    # choose group
    print("Available fixed-parameter groups:")
    keys = list(groups.keys())
    for i, fixed in enumerate(keys, 1):
        desc = ', '.join(f"{k}={v}" for k, v in fixed)
        print(f"{i}. {desc}")
    choice = int(input(f"Select group (1-{len(keys)}): ")) - 1
    selected = groups[keys[choice]]
    selected = sorted(selected, key=lambda x: x[varying])

    # load & plot
    data_list = []
    values = []
    for exp in selected:
        steps, energy, std = load_metrics(exp['log_dir'])
        data_list.append((steps, energy, std))
        values.append(exp[varying])
    plot_experiments(data_list, values, varying)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize VMC experiments by varying a hyperparameter')
    parser.add_argument('base_dir', help='Base directory for TensorBoard logs')
    parser.add_argument('--ansatz', default='GNN2adv', help='Ansatz name to filter')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--output_emb_size', type=int, default=64)
    parser.add_argument('--K_layer', type=int, default=3)
    parser.add_argument('--size', type=int, default=4)
    args = parser.parse_args()
    ansatz_params = {
        'hidden_size': args.hidden_size,
        'output_emb_size': args.output_emb_size,
        'K_layer': args.K_layer
    }
    analyse_varying(args.base_dir, args.ansatz, ansatz_params, args.size)
