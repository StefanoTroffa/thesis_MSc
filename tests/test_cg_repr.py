import unittest
import numpy as np
from compgraph.cg_repr import *
class TestCgRepr(unittest.TestCase):
    def setUp(self) -> None:
        self.config= np.array([-1, -1, 1])
        return super().setUp()
    

    def test_apply_raising_operator(self):
        site = 0
        expected_config = np.array([1, -1, 1])
        new_config = apply_raising_operator(self.config, site)
        self.assertTrue(np.isclose(new_config, expected_config))
        self.assertFalse(np.isclose(self.config, new_config))
        
        # Test that applying the operator to a spin-up state returns None
        new_config = apply_raising_operator(expected_config, site)
        self.assertIsNone(new_config)

    def test_apply_lowering_operator(self):
        site = 0
        expected_config = np.array([-1, -1, 1])
        new_config = apply_lowering_operator(self.config, site)
        self.assertTrue(np.isclose(new_config, expected_config))
        self.assertFalse(np.isclose(self.config, new_config))
        # Test that applying the operator to a spin-down state returns None
        new_config = apply_lowering_operator(new_config, site)
        self.assertIsNone(new_config)


    def test_are_configs_identical(self):
        config1 = np.array([1, -1, 1])
        config2 = np.array([1, -1, 1])
        self.assertTrue(are_configs_identical(config1, config2))
        config3 = np.array([-1, -1, 1])
        self.assertFalse(are_configs_identical(config1, config3))

    def test_configs_differ_by_two_sites(self):
        config1 = np.array([1, -1, 1])
        config2 = np.array([-1, 1, 1])
        self.assertTrue(configs_differ_by_two_sites(config1, config2))
        config3 = np.array([-1, -1, 1])
        self.assertFalse(configs_differ_by_two_sites(config1, config3))
        config4 = np.array([1, -1, -1])
        self.assertFalse(configs_differ_by_two_sites(config1, config4))

if __name__ == '__main__':
    unittest.main()
    