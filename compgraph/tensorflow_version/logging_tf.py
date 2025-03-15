import tensorflow as tf
from datetime import datetime


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
def setup_tensorboard_logging():
    """Set up TensorBoard logging with a unique directory for each run"""
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/vmc_run_{current_time}"
    # log_dir = "logs/vmc_run"
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer, log_dir


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