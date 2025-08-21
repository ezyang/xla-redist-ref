import os
import unittest
import numpy as np

# Set XLA flags before importing JAX
os.environ["XLA_FLAGS"] = " ".join([
    "--xla_force_host_platform_device_count=8",
    "--xla_dump_to=/tmp/xla",
    "--xla_dump_hlo_as_text",
    "--xla_dump_hlo_pass_re=spmd"
])

import jax
from jax.sharding import Mesh, PartitionSpec as P

from mesh_utils import create_device_mesh
from hlo_generation import generate_hlo_for_sharding_constraints
from hlo_analysis import test_hlo_interpreter_and_extract_stablehlo


class Test2D2DPlacements(unittest.TestCase):
    """Test all possible input-output placements for 2-D mesh and 2-D tensor."""
    
    def setUp(self):
        """Set up test environment with 2-D mesh and 2-D tensor shape."""
        self.mesh_shape = (2, 2)  # 2-D mesh with 4 devices (2x2)
        self.array_shape = (8, 8)  # 2-D tensor - both dimensions divisible by 2
        # Create mesh manually like generate_hlo.py does
        devices = np.array(jax.devices()[:4]).reshape(2, 2)
        self.mesh = Mesh(devices, ('a', 'b'))
    
    def _test_2d_2d(self, in_spec: P, out_spec: P) -> bool:
        """Helper function to test a specific input-output placement combination."""
        hlo = generate_hlo_for_sharding_constraints(
            in_spec, out_spec, self.mesh_shape, self.array_shape
        )
        
        is_identity, _, _ = test_hlo_interpreter_and_extract_stablehlo(
            hlo, self.mesh, in_spec, out_spec, self.array_shape
        )
        
        return is_identity
    
    # Test all combinations of input and output specs for 2D mesh with 2D tensor
    # For 2D tensor with 2D mesh (a, b), possible specs are:
    # - P(None, None): replicated
    # - P('a', None): sharded along first dimension on mesh axis 'a'
    # - P(None, 'a'): sharded along second dimension on mesh axis 'a'
    # - P('b', None): sharded along first dimension on mesh axis 'b'
    # - P(None, 'b'): sharded along second dimension on mesh axis 'b'
    # - P('a', 'b'): sharded along both dimensions
    # - P('b', 'a'): sharded along both dimensions (swapped)
    
    def test_replicated_to_replicated(self):
        """Test P(None, None) -> P(None, None)"""
        self.assertTrue(self._test_2d_2d(P(None, None), P(None, None)))
    
    def test_replicated_to_sharded_a_none(self):
        """Test P(None, None) -> P('a', None)"""
        self.assertTrue(self._test_2d_2d(P(None, None), P('a', None)))
    
    def test_replicated_to_sharded_none_a(self):
        """Test P(None, None) -> P(None, 'a')"""
        self.assertTrue(self._test_2d_2d(P(None, None), P(None, 'a')))
    
    def test_replicated_to_sharded_b_none(self):
        """Test P(None, None) -> P('b', None)"""
        self.assertTrue(self._test_2d_2d(P(None, None), P('b', None)))
    
    def test_replicated_to_sharded_none_b(self):
        """Test P(None, None) -> P(None, 'b')"""
        self.assertTrue(self._test_2d_2d(P(None, None), P(None, 'b')))
    
    def test_replicated_to_sharded_a_b(self):
        """Test P(None, None) -> P('a', 'b')"""
        self.assertTrue(self._test_2d_2d(P(None, None), P('a', 'b')))
    
    def test_replicated_to_sharded_b_a(self):
        """Test P(None, None) -> P('b', 'a')"""
        self.assertTrue(self._test_2d_2d(P(None, None), P('b', 'a')))
    
    def test_sharded_a_none_to_replicated(self):
        """Test P('a', None) -> P(None, None)"""
        self.assertTrue(self._test_2d_2d(P('a', None), P(None, None)))
    
    def test_sharded_a_none_to_sharded_a_none(self):
        """Test P('a', None) -> P('a', None)"""
        self.assertTrue(self._test_2d_2d(P('a', None), P('a', None)))
    
    def test_sharded_a_none_to_sharded_none_a(self):
        """Test P('a', None) -> P(None, 'a')"""
        self.assertTrue(self._test_2d_2d(P('a', None), P(None, 'a')))
    
    def test_sharded_a_none_to_sharded_b_none(self):
        """Test P('a', None) -> P('b', None)"""
        self.assertTrue(self._test_2d_2d(P('a', None), P('b', None)))
    
    def test_sharded_a_none_to_sharded_none_b(self):
        """Test P('a', None) -> P(None, 'b')"""
        self.assertTrue(self._test_2d_2d(P('a', None), P(None, 'b')))
    
    def test_sharded_a_none_to_sharded_a_b(self):
        """Test P('a', None) -> P('a', 'b')"""
        self.assertTrue(self._test_2d_2d(P('a', None), P('a', 'b')))
    
    def test_sharded_a_none_to_sharded_b_a(self):
        """Test P('a', None) -> P('b', 'a')"""
        self.assertTrue(self._test_2d_2d(P('a', None), P('b', 'a')))
    
    def test_sharded_none_a_to_replicated(self):
        """Test P(None, 'a') -> P(None, None)"""
        self.assertTrue(self._test_2d_2d(P(None, 'a'), P(None, None)))
    
    def test_sharded_none_a_to_sharded_a_none(self):
        """Test P(None, 'a') -> P('a', None)"""
        self.assertTrue(self._test_2d_2d(P(None, 'a'), P('a', None)))
    
    def test_sharded_none_a_to_sharded_none_a(self):
        """Test P(None, 'a') -> P(None, 'a')"""
        self.assertTrue(self._test_2d_2d(P(None, 'a'), P(None, 'a')))
    
    def test_sharded_none_a_to_sharded_b_none(self):
        """Test P(None, 'a') -> P('b', None)"""
        self.assertTrue(self._test_2d_2d(P(None, 'a'), P('b', None)))
    
    def test_sharded_none_a_to_sharded_none_b(self):
        """Test P(None, 'a') -> P(None, 'b')"""
        self.assertTrue(self._test_2d_2d(P(None, 'a'), P(None, 'b')))
    
    def test_sharded_none_a_to_sharded_a_b(self):
        """Test P(None, 'a') -> P('a', 'b')"""
        self.assertTrue(self._test_2d_2d(P(None, 'a'), P('a', 'b')))
    
    def test_sharded_none_a_to_sharded_b_a(self):
        """Test P(None, 'a') -> P('b', 'a')"""
        self.assertTrue(self._test_2d_2d(P(None, 'a'), P('b', 'a')))
    
    def test_sharded_b_none_to_replicated(self):
        """Test P('b', None) -> P(None, None)"""
        self.assertTrue(self._test_2d_2d(P('b', None), P(None, None)))
    
    def test_sharded_b_none_to_sharded_a_none(self):
        """Test P('b', None) -> P('a', None)"""
        self.assertTrue(self._test_2d_2d(P('b', None), P('a', None)))
    
    def test_sharded_b_none_to_sharded_none_a(self):
        """Test P('b', None) -> P(None, 'a')"""
        self.assertTrue(self._test_2d_2d(P('b', None), P(None, 'a')))
    
    def test_sharded_b_none_to_sharded_b_none(self):
        """Test P('b', None) -> P('b', None)"""
        self.assertTrue(self._test_2d_2d(P('b', None), P('b', None)))
    
    def test_sharded_b_none_to_sharded_none_b(self):
        """Test P('b', None) -> P(None, 'b')"""
        self.assertTrue(self._test_2d_2d(P('b', None), P(None, 'b')))
    
    def test_sharded_b_none_to_sharded_a_b(self):
        """Test P('b', None) -> P('a', 'b')"""
        self.assertTrue(self._test_2d_2d(P('b', None), P('a', 'b')))
    
    def test_sharded_b_none_to_sharded_b_a(self):
        """Test P('b', None) -> P('b', 'a')"""
        self.assertTrue(self._test_2d_2d(P('b', None), P('b', 'a')))
    
    def test_sharded_none_b_to_replicated(self):
        """Test P(None, 'b') -> P(None, None)"""
        self.assertTrue(self._test_2d_2d(P(None, 'b'), P(None, None)))
    
    def test_sharded_none_b_to_sharded_a_none(self):
        """Test P(None, 'b') -> P('a', None)"""
        self.assertTrue(self._test_2d_2d(P(None, 'b'), P('a', None)))
    
    def test_sharded_none_b_to_sharded_none_a(self):
        """Test P(None, 'b') -> P(None, 'a')"""
        self.assertTrue(self._test_2d_2d(P(None, 'b'), P(None, 'a')))
    
    def test_sharded_none_b_to_sharded_b_none(self):
        """Test P(None, 'b') -> P('b', None)"""
        self.assertTrue(self._test_2d_2d(P(None, 'b'), P('b', None)))
    
    def test_sharded_none_b_to_sharded_none_b(self):
        """Test P(None, 'b') -> P(None, 'b')"""
        self.assertTrue(self._test_2d_2d(P(None, 'b'), P(None, 'b')))
    
    def test_sharded_none_b_to_sharded_a_b(self):
        """Test P(None, 'b') -> P('a', 'b')"""
        self.assertTrue(self._test_2d_2d(P(None, 'b'), P('a', 'b')))
    
    def test_sharded_none_b_to_sharded_b_a(self):
        """Test P(None, 'b') -> P('b', 'a')"""
        self.assertTrue(self._test_2d_2d(P(None, 'b'), P('b', 'a')))
    
    def test_sharded_a_b_to_replicated(self):
        """Test P('a', 'b') -> P(None, None)"""
        self.assertTrue(self._test_2d_2d(P('a', 'b'), P(None, None)))
    
    def test_sharded_a_b_to_sharded_a_none(self):
        """Test P('a', 'b') -> P('a', None)"""
        self.assertTrue(self._test_2d_2d(P('a', 'b'), P('a', None)))
    
    def test_sharded_a_b_to_sharded_none_a(self):
        """Test P('a', 'b') -> P(None, 'a')"""
        self.assertTrue(self._test_2d_2d(P('a', 'b'), P(None, 'a')))
    
    def test_sharded_a_b_to_sharded_b_none(self):
        """Test P('a', 'b') -> P('b', None)"""
        self.assertTrue(self._test_2d_2d(P('a', 'b'), P('b', None)))
    
    def test_sharded_a_b_to_sharded_none_b(self):
        """Test P('a', 'b') -> P(None, 'b')"""
        self.assertTrue(self._test_2d_2d(P('a', 'b'), P(None, 'b')))
    
    def test_sharded_a_b_to_sharded_a_b(self):
        """Test P('a', 'b') -> P('a', 'b')"""
        self.assertTrue(self._test_2d_2d(P('a', 'b'), P('a', 'b')))
    
    def test_sharded_a_b_to_sharded_b_a(self):
        """Test P('a', 'b') -> P('b', 'a')"""
        self.assertTrue(self._test_2d_2d(P('a', 'b'), P('b', 'a')))
    
    def test_sharded_b_a_to_replicated(self):
        """Test P('b', 'a') -> P(None, None)"""
        self.assertTrue(self._test_2d_2d(P('b', 'a'), P(None, None)))
    
    def test_sharded_b_a_to_sharded_a_none(self):
        """Test P('b', 'a') -> P('a', None)"""
        self.assertTrue(self._test_2d_2d(P('b', 'a'), P('a', None)))
    
    def test_sharded_b_a_to_sharded_none_a(self):
        """Test P('b', 'a') -> P(None, 'a')"""
        self.assertTrue(self._test_2d_2d(P('b', 'a'), P(None, 'a')))
    
    def test_sharded_b_a_to_sharded_b_none(self):
        """Test P('b', 'a') -> P('b', None)"""
        self.assertTrue(self._test_2d_2d(P('b', 'a'), P('b', None)))
    
    def test_sharded_b_a_to_sharded_none_b(self):
        """Test P('b', 'a') -> P(None, 'b')"""
        self.assertTrue(self._test_2d_2d(P('b', 'a'), P(None, 'b')))
    
    def test_sharded_b_a_to_sharded_a_b(self):
        """Test P('b', 'a') -> P('a', 'b')"""
        self.assertTrue(self._test_2d_2d(P('b', 'a'), P('a', 'b')))
    
    def test_sharded_b_a_to_sharded_b_a(self):
        """Test P('b', 'a') -> P('b', 'a')"""
        self.assertTrue(self._test_2d_2d(P('b', 'a'), P('b', 'a')))


if __name__ == '__main__':
    unittest.main()