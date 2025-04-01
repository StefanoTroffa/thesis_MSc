import tensorflow as tf
from datetime import datetime
import os
import json
def setup_tensorboard_model_weights(model, sample_input):
    log_dir = "logs_example/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(log_dir)
    print(f"loading data to {log_dir}")
    # Simple graph tracing without profiler
    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)
    _ = model(sample_input)
    with writer.as_default():
        tf.summary.trace_export(
            name="weights",
            step=0
        )
    return writer, log_dir

def log_weights(step, model, writer):
    """Safe weight logging after model initialization"""
    try:
        with writer.as_default():
            for var in model.trainable_variables:
                tf.summary.histogram(f"weights/{var.name}", var, step=step)
                tf.summary.scalar(f"nan_count/{var.name}", 
                                tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32)), 
                                step=step)
    except ValueError as e:
        print(f"Logging skipped: {str(e)}")
        tf.summary.text("errors/weight_logging", str(e), step=step)
    return
def setup_tensorboard_logging(location:str='vmc_run'):
    """Set up TensorBoard logging with a unique directory for each run"""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{location}_{current_time}"
    # log_dir = "logs/vmc_run"
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer, log_dir

def setup_tensorboard_loggingv2(hyperparams, base_dir="tensorboard_logs"):
    """Set up TensorBoard logging with directory structure matching hyperparameters
    
    Args:
        hyperparams (Hyperams): Dataclass containing simulation parameters
        base_dir (str): Base directory for tensorboard logs
        
    Returns:
        tuple: (summary_writer, log_dir) - TensorBoard writer and log directory path
    """


    # Extract all parameters from the dataclasses
    graph_params = hyperparams.graph_params
    sim_params = hyperparams.sim_params
    
    # Create directory components
    graph_info = f"{graph_params.graphType}_{graph_params.n:02d}_{graph_params.m:02d}_{graph_params.sublattice}"
    
    # Format simulation parameters in a consistent way
    sim_info = f"beta{sim_params.beta}__bs_{sim_params.batch_size}lr{sim_params.learning_rate:.1e}_loop{sim_params.outer_loop}x{sim_params.inner_loop}_{sim_params.gradient}"
    
    # Format ansatz information
    ansatz_info = f"{hyperparams.ansatz}_h{hyperparams.ansatz_params.get('hidden_size', 0)}_e{hyperparams.ansatz_params.get('output_emb_size', 0)}"
    
    # Include simulation type (VMC, ExactVMC, etc.)
    sim_type = hyperparams.symulation_type
    
    # Combine components into path - using job ID instead of datetime for uniqueness
    log_dir = os.path.join(
        base_dir,
        f"system_Heisenberg",
        graph_info,
        f"{sim_info}_{sim_type}",
        ansatz_info
    )
    
    # Create directory
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Create the summary writer
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Log hyperparameters as text for reference
    with summary_writer.as_default():
        # Convert dataclasses to dictionaries for logging
        params_dict = {
            "simulation_type": hyperparams.symulation_type,
            "graph_params": {k: v for k, v in vars(hyperparams.graph_params).items()},
            "sim_params": {k: v for k, v in vars(hyperparams.sim_params).items()},
            "ansatz": hyperparams.ansatz,
            "ansatz_params": hyperparams.ansatz_params
        }
        
        # Log as formatted text
        param_text = json.dumps(params_dict, indent=2)
        tf.summary.text('hyperparameters/config', param_text, step=0)
        
        # Also log individual parameters as scalars for plotting/tracking
        for param_name, param_value in params_dict["sim_params"].items():
            if isinstance(param_value, (int, float)):
                tf.summary.scalar(f'hyperparameters/{param_name}', param_value, step=0)
    
    return summary_writer, log_dir


def extract_hyperparams_from_path(log_dir_path):
    """
    Extract hyperparameters from a tensorboard log directory path.
    
    Parameters:
    -----------
    log_dir_path : str
        Path to the tensorboard log directory
        
    Returns:
    --------
    dict : Dictionary of extracted hyperparameters
    """
    # Extract the base path components
    path_components = log_dir_path.split('/')
    
    # Extract system type
    system_type = None
    for component in path_components:
        if component.startswith('system_'):
            system_type = component.replace('system_', '')
    
    # Extract lattice information from the component after system type
    lattice_info = None
    for i, component in enumerate(path_components):
        if component.startswith('system_') and i+1 < len(path_components):
            lattice_info = path_components[i+1]
            break
    
    # Parse lattice type and dimensions
    lattice_type = None
    n = None
    m = None
    initialization = None
    
    if lattice_info:
        # Split by underscore
        lattice_parts = lattice_info.split('_')
        
        # First part is the lattice type (e.g., '2dsquare')
        if len(lattice_parts) > 0:
            lattice_type = lattice_parts[0]
        
        # Get dimensions if available
        if len(lattice_parts) > 2:
            try:
                n = int(lattice_parts[1])
                m = int(lattice_parts[2])
            except ValueError:
                pass
        
        # Check if there's initialization info (e.g., 'Neel')
        if len(lattice_parts) > 3:
            initialization = lattice_parts[3]
    
    # Extract hyperparameters from the final component
    hyperparams_component = path_components[-1]
    hyperparams_parts = hyperparams_component.split('_')
    
    beta = None
    learning_rate = None
    outer_loop = None
    inner_loop = None
    gradient_type = None
    model_type = None
    
    for part in hyperparams_parts:
        if part.startswith('beta'):
            try:
                beta = float(part.replace('beta', ''))
            except ValueError:
                pass
        elif part.startswith('lr'):
            try:
                learning_rate = float(part.replace('lr', ''))
            except ValueError:
                pass
        elif 'x' in part and part[0].isdigit() and part[-1].isdigit():
            # Extract loop values (e.g., '500x30')
            loop_parts = part.split('x')
            if len(loop_parts) == 2:
                try:
                    outer_loop = int(loop_parts[0])
                    inner_loop = int(loop_parts[1])
                except ValueError:
                    pass
        elif part in ['energy', 'overlap']:
            gradient_type = part
        elif part in ['ExactVMC']:
            model_type = part
    
    # Compile all extracted information
    hyperparams = {
        'system_type': system_type,
        'lattice': {
            'type': lattice_type,
            'n': n,
            'm': m,
            'initialization': initialization
        },
        'simulation': {
            'beta': beta,
            'learning_rate': learning_rate,
            'outer_loop': outer_loop,
            'inner_loop': inner_loop,
            'gradient_type': gradient_type,
            'model_type': model_type
        }
    }
    
    return hyperparams


def log_gradient_norms(step, gradients, writer):
    """
    Log gradient norms for each trainable variable.
    Software: Helps detect numerical instabilities (exploding/vanishing gradients).
    Hardware: Provides insight into compute load and potential GPU/CPU stress.
    """
    with writer.as_default():
        for i, grad in enumerate(gradients):
            if grad is not None:
                norm = tf.norm(grad)
                tf.summary.scalar(f'gradients/norm_{i}', norm, step=step)
    return                

def log_training_metrics(summary_writer, step, metrics_dict):
    """Log training metrics to TensorBoard"""
    with summary_writer.as_default():
        tf.summary.scalar('training/energy', metrics_dict['energy'], step=step)
        if 'magnetization' in metrics_dict:
            tf.summary.scalar('training/magnetization', metrics_dict['magnetization'], step=step)
        if 'overlap' in metrics_dict:
            tf.summary.scalar('training/overlap', metrics_dict['overlap'], step=step)
        tf.summary.scalar('memory/ram_used_mb', metrics_dict['ram_used_mb'], step=step)
        if 'gpu_memory_mb' in metrics_dict:
            tf.summary.scalar('memory/gpu_used_mb', metrics_dict['gpu_memory_mb'], step=step)
        if 'learning_rate' in metrics_dict:
            tf.summary.scalar('training/learning_rate', metrics_dict['learning_rate'], step=step)
        if 'notes' in metrics_dict:
            tf.summary.text('notes/custom_message', metrics_dict['notes'], step=step)

def log_weights_and_nan_check(step, model, writer):
    """
    Log model weight histograms and the count of NaNs.
    Software: Helps debug weight divergence or accumulation of NaNs.
    Hardware: Assists in monitoring memory usage and precision issues on GPU/CPU.
    """
    with writer.as_default():
        for var in model.trainable_variables:
            tf.summary.histogram(f"weights/{var.name}", var, step=step)
            nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(var), tf.int32))
            tf.summary.scalar(f"nan_count/{var.name}", nan_count, step=step)
            zero_count = tf.reduce_sum(tf.cast(tf.equal(var, 0.0), tf.int32))
            tf.summary.scalar(f"zero_count/{var.name}", zero_count, step=step)

def initialize_checkpoint(log_dir, model_var, optimizer):
    """
    Initialize TensorFlow checkpointing for model and optimizer.
    """
    # Create checkpoint directory within the log directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint manager for regular checkpoints
    checkpoint = tf.train.Checkpoint(model=model_var, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, 
        directory=checkpoint_dir,
        max_to_keep=5  # Keep multiple checkpoints
    )
    
    # Save initial checkpoint separately with a special name
    initial_checkpoint = tf.train.Checkpoint(model=model_var, optimizer=optimizer)
    initial_checkpoint_path = os.path.join(checkpoint_dir, "initial_model")
    initial_checkpoint.save(initial_checkpoint_path)
    print(f"Initial model checkpoint saved to: {initial_checkpoint_path}")
    return checkpoint_manager