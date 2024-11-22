import unittest
import numpy as np
import quimb as qu
import networkx as nx
from compgraph.models import GNN_double_output, GNN_double_output_advanced
from compgraph.tensor_wave_functions import variational_wave_function_on_batch, sparse_tensor_exp_energy, create_sparsetensor_from_configs_amplitudes, time_evoluted_wave_function_on_batch, montecarlo_logloss_overlap_time_evoluted, calculate_sparse_overlap, quimb_vec_to_sparse
import tensorflow as tf
from simulation.initializer import initialize_NQS_model_fromhyperparams
class TestModelInitialization(unittest.TestCase):
    def test_successful_initialization_simple(self):
        """ Test initialization of GNN_double_output model. """
        hyperparams = {
            'ansatz': 'GNN2simple',
            'ansatz_params': {
                'hidden_size': 8,
                'output_emb_size': 12
            }
        }
        model = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'], hyperparams['ansatz_params'])
        self.assertIsInstance(model, GNN_double_output)

    def test_successful_initialization_advanced(self):
        """ Test initialization of GNN_double_output_advanced model. """
        hyperparams = {
            'ansatz': 'GNN2adv',
            'ansatz_params': {
                'K_layer': 3,
                'hidden_size': 8,
                'output_emb_size': 12
            }
        }
        model = initialize_NQS_model_fromhyperparams(hyperparams['ansatz'], hyperparams['ansatz_params'])
        self.assertIsInstance(model, GNN_double_output_advanced)

    def test_failure_initialization(self):
        """ Test failure of model initialization for unsupported ansatz type. """
        hyperparams = {
            'ansatz': 'unknown_model',
            'ansatz_params': {
                'K_layer': 3,
                'hidden_size': 8,
                'output_emb_size': 12
            }
        }
        with self.assertRaises(ValueError) as context:
            initialize_NQS_model_fromhyperparams(hyperparams['ansatz'], hyperparams['ansatz_params'])
        self.assertIn('This model cannot be initialized. The available models are:', str(context.exception))

if __name__ == '__main__':
    unittest.main()