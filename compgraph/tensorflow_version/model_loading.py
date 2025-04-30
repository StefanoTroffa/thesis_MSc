import tensorflow as tf
from simulation.initializer import initialize_NQS_model_fromhyperparams
def load_model_from_path(model, checkpoint_path, optimizer):
    """
    Load model weights from a checkpoint.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint
    """
    
    # Restore the model
    new_checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    status = new_checkpoint.restore(checkpoint_path)
    
    # Assert that all variables were restored
    status.assert_existing_objects_matched()

import re
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(frozen=True)
class GraphParams:
    graphType: str = "2dsquare"
    n: int = 2
    m: int = 3
    sublattice: str = "Neel"

@dataclass(frozen=True)
class SimParams:
    beta: float = 0.2
    batch_size: int = 128
    learning_rate: float = 7e-5  
    outer_loop: int = 256
    inner_loop: int = 18
    gradient: str = 'overlap'

@dataclass
class Hyperparams:
    simulation_type: str = "VMC"
    graph_params: GraphParams = field(default_factory=GraphParams)
    sim_params: SimParams = field(default_factory=SimParams)
    ansatz: str = "GNN2adv"
    ansatz_params: dict = field(default_factory=lambda: {"hidden_size": 128, "output_emb_size": 64, 'K_layer': 2})

def extract_hyperparams_from_path(path: str) -> Hyperparams:
    """
    Extract hyperparameters from a checkpoint path.
    
    Example path:
    '/home/s3378209/data1/repfin/thesis_MSc/checkpointed_logs/system_Heisenberg/2dsquare_03_03_Neel/beta0.07__bs_128lr7.0e-05_loop256x18_overlap_VMC/GNN2adv_h128_e64_K2/checkpoints/ckpt-9'
    
    Returns:
        Hyperparams: A dataclass containing the extracted hyperparameters
    """
    # Extract graph parameters
    graph_params_dict = {}
    graph_match = re.search(r'/(2dsquare)_(\d+)_(\d+)_(\w+)/', path)
    if graph_match:
        graph_params_dict["graphType"] = graph_match.group(1)
        graph_params_dict["n"] = int(graph_match.group(2))
        graph_params_dict["m"] = int(graph_match.group(3))
        graph_params_dict["sublattice"] = graph_match.group(4)
    
    # Extract simulation parameters
    sim_params_dict = {}
    simulation_type = "VMC"  # Default value
    sim_match = re.search(r'/beta(\d+\.\d+)__bs_(\d+)lr([\d.e-]+)_loop(\d+)x(\d+)_(\w+)_(\w+)/', path)
    if sim_match:
        sim_params_dict["beta"] = float(sim_match.group(1))
        sim_params_dict["batch_size"] = int(sim_match.group(2))
        sim_params_dict["learning_rate"] = float(sim_match.group(3))
        sim_params_dict["outer_loop"] = int(sim_match.group(4))
        sim_params_dict["inner_loop"] = int(sim_match.group(5))
        sim_params_dict["gradient"] = sim_match.group(6)
        simulation_type = sim_match.group(7)
    
    # Extract ansatz parameters
    ansatz = "GNN2adv"  # Default value
    ansatz_params = {}
    ansatz_match = re.search(r'/(\w+)_h(\d+)_e(\d+)_K(\d+)/', path)
    if ansatz_match:
        ansatz = ansatz_match.group(1)
        ansatz_params["hidden_size"] = int(ansatz_match.group(2))
        ansatz_params["output_emb_size"] = int(ansatz_match.group(3))
        ansatz_params["K_layer"] = int(ansatz_match.group(4))
    
    # Create the dataclass instances with the extracted parameters
    graph_params = GraphParams(**graph_params_dict) if graph_params_dict else GraphParams()
    sim_params = SimParams(**sim_params_dict) if sim_params_dict else SimParams()
    
    # Create and return the Hyperparams instance
    return Hyperparams(
        simulation_type=simulation_type,
        graph_params=graph_params,
        sim_params=sim_params,
        ansatz=ansatz,
        ansatz_params=ansatz_params
    )

# Update the model checking function
def check_and_reinitialize_model(model, GT_Batch, hyperparams, tolerance=0.01, max_attempts=5, seed=None):
    """
    Check if all outputs from the model on the batch are too similar (within tolerance range).
    If so, reinitialize the model until we get diverse outputs or hit max attempts.
    
    Args:
        model: The neural network model
        GT_Batch: Graph tuple batch input
        hyperparams: Hyperparameters object
        tolerance: Allowed variation between outputs (as a fraction, default 0.01 or 1%)
        max_attempts: Maximum number of reinitialization attempts
        
    Returns:
        model: Either the original model or a reinitialized one
    """
    # Get model outputs on the batch
    outputs = model(GT_Batch)
    
    # Calculate mean output and check variation
    mean_output = tf.reduce_mean(outputs[:,0]), tf.reduce_mean(outputs[:,1])
    std_output = tf.math.reduce_std(outputs[:,0]), tf.math.reduce_std(outputs[:,1])

    print("Ciao first trial seconmd")
    attempt = 0
    print(f"Initial std: {std_output[0]:.6f} (tolerance: {tolerance}), attempt: {attempt}/{max_attempts},{(tf.reduce_sum(std_output)) < tolerance}")
    
    # Keep reinitializing until we get diverse outputs or hit max attempts
    while (std_output[0] < tolerance or std_output[1] < tolerance) and attempt < max_attempts:
        attempt += 1
        
        # Generate a new random seed for this initialization
        seed = tf.random.uniform([], minval=0, maxval=1000000, dtype=tf.int32)
        print(f"WARNING: Model outputs too similar {std_output[0]:.6f} Reinitializing model with seed {seed} (attempt {attempt}/{max_attempts})...")
        
        # Delete the current model to free memory
        del model
        
        # Reinitialize the model with a new random seed
        model_new = initialize_NQS_model_fromhyperparams(
            hyperparams.ansatz, 
            hyperparams.ansatz_params,
            seed=seed
        )
        
        # Initialize the new model with a forward pass
        model_new(GT_Batch)
        print(model_new(GT_Batch)[0])
        
        # Replace the old model
        model = model_new
        
        # Free memory
        del model_new
        
        # Check again
        outputs = model(GT_Batch)
        std_output = tf.math.reduce_std(outputs[:,0]), tf.math.reduce_std(outputs[:,1])

    if attempt > 0:
        if tf.reduce_sum(std_output) < tolerance:
            print(f"WARNING: Model still producing too similar outputs after {attempt} reinitialization attempts! (std deviation: {std_output[0]:.6f}, {std_output[1]:.6f})")
        else:
            print(f"Model successfully reinitialized with diverse outputs. (std deviation: {tf.reduce_sum(std_output):.6f})")
    seed_val= seed if tf.reduce_sum(std_output) > tolerance else -1
    return model, seed_val
def explore_model(model):
    """
    Explore the variable structure of a Sonnet model.
    
    Args:
        model: A Sonnet model
    """

    
    print(f"Total variables in model: {len(model.variables)}")
    
    # Group variables by module
    modules = {}
    for var in model.variables:
        name = var.name
        parts = name.split('/')
        if len(parts) >= 2:
            module = parts[0]
            if module not in modules:
                modules[module] = []
            modules[module].append(var)
    
    # Print module hierarchy
    print("\nModule hierarchy:")
    for module, vars in modules.items():
        print(f"  {module} ({len(vars)} variables)")
        
        # Group by submodule
        submodules = {}
        for var in vars:
            name = var.name
            parts = name.split('/')
            if len(parts) >= 3:
                submodule = parts[1]
                if submodule not in submodules:
                    submodules[submodule] = []
                submodules[submodule].append(var)
        
        # Print submodules
        for submodule, subvars in submodules.items():
            print(f"    {submodule} ({len(subvars)} variables)")
            
            # Print a few example variables
            for i, var in enumerate(subvars):
                print(f"      {var.name}: {var.shape}")
          
