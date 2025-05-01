#!/usr/bin/env python3
import os
import argparse
import socket                 # ← added
import subprocess
from datetime import datetime

# ———————— CLI FLAGS ——————————————————————————————————
parser = argparse.ArgumentParser(
    description="Profile Monte Carlo grid: batch×system, power, mem, timing"
)
parser.add_argument("--cpu", action="store_true",
                    help="Disable GPU; force CPU-only execution")
parser.add_argument("--run-eagerly", action="store_true",
                    help="Force eager execution (no tf.function graphs)")
args = parser.parse_args()

# ————— PRE-TF IMPORT CONFIG ———————————————————————————
if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
# Must come before importing any @tf.function-decorated code:
tf.config.run_functions_eagerly(bool(args.run_eagerly))
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# ————— DEPENDENCIES —————————————————————————————————
import time, gc, psutil, pandas as pd
from compgraph.monte_carlo import MCMCSampler, compute_phi_terms
from compgraph.tensorflow_version.hamiltonian_operations import (
    stochastic_energy_tf, stochastic_gradients_tfv3
)
from compgraph.tensorflow_version.memory_control import aggressive_memory_cleanup
from simulation.initializer import (
    create_graph_from_ham, initialize_NQS_model_fromhyperparams
)
from compgraph.tensorflow_version.graph_tuple_manipulation import (
    initialize_graph_tuples_tf_opt, precompute_graph_structure
)
from compgraph.useful import copy_to_non_trainable
import sonnet as snt

# ————— nvidia-smi HELPERS —————————————————————————————
def query_nvidia_smi(field: str) -> float:
    out = subprocess.check_output([
        "nvidia-smi", f"--query-gpu={field}",
        "--format=csv,noheader,nounits"
    ], encoding="utf-8").split("\n",1)[0]
    try:   return float(out)
    except: return 0.0

def sample_power():    return query_nvidia_smi("power.draw")
def sample_gpu_util(): return query_nvidia_smi("utilization.gpu")
def monitor_gpu_mem(): return query_nvidia_smi("memory.used")

# ————— CORE PROFILING —————————————————————————————————
def track_all_operations(*, n_iterations=20, batch_size=32, lattice_n=6):
    # 1) Reset & cleanup
    tf.keras.backend.clear_session()
    aggressive_memory_cleanup()

    # 2) Decide sublattice encoding
    sublattice = "Neel" if lattice_n % 2 == 0 else "Alternatepattern"

    # 3) Build graph & model
    graph, subl = create_graph_from_ham("2dsquare",
                                        (lattice_n, lattice_n),
                                        sublattice)
    params = {"hidden_size":128, "output_emb_size":64, "K_layer":2}
    model_w = initialize_NQS_model_fromhyperparams("GNNprocnorm", params, seed=860432)
    model_fix = initialize_NQS_model_fromhyperparams("GNNprocnorm", params)
    GT = initialize_graph_tuples_tf_opt(batch_size, graph, subl)
    _, _, edge_pairs = precompute_graph_structure(graph)
    model_w(GT); model_fix(GT)
    copy_to_non_trainable(model_w, model_fix)
    template = initialize_graph_tuples_tf_opt(edge_pairs.shape[0]+1,
                                              graph, subl)

    sampler_var = MCMCSampler(GT, 0.005, edge_pairs=edge_pairs)
    sampler_te  = MCMCSampler(GT, template=template,
                              beta=0.005, edge_pairs=edge_pairs)
    optimizer   = snt.optimizers.Adam(1e-4)
    process     = psutil.Process()

    # 4) Warm-up (unlogged compile/JIT)
    _ = sampler_var.monte_carlo_update_on_batch_profilemem(model_w, GT)

    records = []
    for i in range(1, n_iterations+1):
        gc.collect()
        power_acc = 0.0

        # — MCMC updates —
        ram0 = process.memory_info().rss / 1e6
        t0   = time.time()
        for _ in range(lattice_n * lattice_n):
            GT, psi_new = sampler_var.monte_carlo_update_on_batch_profilemem(
                model_w, GT
            )
        t_mcmc = time.time() - t0
        ram1   = process.memory_info().rss / 1e6

        # — φ-terms —
        t_phi = time.time()
        phi   = compute_phi_terms(GT, sampler_te, model_fix)
        t_phi = time.time() - t_phi

        # — energy + power —
        t_eng  = time.time()
        p0     = sample_power()
        E,std,_= stochastic_energy_tf(psi_new, model_w,
                                      edge_pairs, template, GT, 0.0)
        dt     = time.time() - t_eng; power_acc += p0*dt
        t_eng  = dt

        # — gradients —
        t_grad = time.time()
        loss, grads = stochastic_gradients_tfv3(phi, GT, model_w)
        t_grad = time.time() - t_grad

        # — optimizer apply —
        t_opt = time.time()
        optimizer.apply(grads, model_w.trainable_variables)
        t_opt = time.time() - t_opt

        # — GPU stats —
        gpu_mem  = monitor_gpu_mem()
        gpu_util = sample_gpu_util()

        records.append({
            "batch_size": batch_size,
            "lattice_n":  lattice_n,
            "iteration":  i,
            "time_mcmc_s":   t_mcmc,
            "time_phi_s":    t_phi,
            "time_energy_s": t_eng,
            "time_grad_s":   t_grad,
            "time_opt_s":    t_opt,
            "ram_start_MB":  ram0,
            "ram_end_MB":    ram1,
            "gpu_mem_MB":    gpu_mem,
            "gpu_util_%":    gpu_util,
            "energy_J_per_sample": power_acc/(batch_size*n_iterations),
        })

    return records

# ————— MAIN DRIVER —————————————————————————————————
if __name__ == "__main__":
    all_recs = []
    for n in (3,4,5,6):
        for bs in (32,64,128,256,512):
            print(f"▶ Profile {n}×{n}, batch={bs}  "
                  f"(eager={args.run_eagerly}, cpu={args.cpu})")
            recs = track_all_operations(
                n_iterations=30,
                batch_size=bs,
                lattice_n=n
            )
            all_recs.extend(recs)

            # ◉◉◉ Print summary of what just ran:
            print(f"  • Stored {len(recs)} records for {n}×{n}, bs={bs}")
            print(f"    Last record: {recs[-1]}\n")

    # — Filename metadata —
    node    = socket.gethostname()
    raw_gpu = subprocess.check_output([
        "nvidia-smi","--query-gpu=name","--format=csv,noheader,nounits"
    ], encoding="utf-8").split("\n",1)[0]
    gpu     = raw_gpu.replace(" ","_").replace("/","_")
    ts      = datetime.now().strftime("%Y%m%d-%H%M%S")

    # — Write out CSV + PKL —
    df      = pd.DataFrame(all_recs)
    out_csv = f"grid_{node}_{gpu}_{ts}.csv"
    out_pkl = f"grid_{node}_{gpu}_{ts}.pkl"
    df.to_csv(out_csv, index=False)
    df.to_pickle(out_pkl)

    print(f"✔ Saved results → {out_csv}, {out_pkl}")
