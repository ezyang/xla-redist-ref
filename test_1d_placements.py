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


class Test1DPlacements(unittest.TestCase):
    """Test all possible input-output placements for 1-D mesh and 1-D tensor."""
    
    def setUp(self):
        """Set up test environment with 1-D mesh and tensor shape."""
        self.mesh_shape = (4,)  # 1-D mesh with 4 devices
        self.array_shape = (8,)  # 1-D tensor
        # Create mesh manually like generate_hlo.py does
        devices = np.array(jax.devices()[:4])
        self.mesh = Mesh(devices, ('a',))
    
    def _test_1d_1d(self, in_spec: P, out_spec: P) -> bool:
        """Helper function to test a specific input-output placement combination."""
        hlo = generate_hlo_for_sharding_constraints(
            in_spec, out_spec, self.mesh_shape, self.array_shape
        )
        
        is_identity, _, _ = test_hlo_interpreter_and_extract_stablehlo(
            hlo, self.mesh, in_spec, out_spec, self.array_shape
        )
        
        return is_identity
    
    def test_replicated_to_replicated(self):
        """Test P(None) -> P(None)"""
        self.assertTrue(self._test_1d_1d(P(None), P(None)))
    
    def test_replicated_to_sharded_a(self):
        """Test P(None) -> P('a')"""
        self.assertTrue(self._test_1d_1d(P(None), P('a')))
    
    def test_sharded_a_to_replicated(self):
        """Test P('a') -> P(None)"""
        self.assertTrue(self._test_1d_1d(P('a'), P(None)))
    
    def test_sharded_a_to_sharded_a(self):
        """Test P('a') -> P('a')"""
        self.assertTrue(self._test_1d_1d(P('a'), P('a')))


if __name__ == '__main__':
    unittest.main()