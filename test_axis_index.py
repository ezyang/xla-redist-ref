import os
import numpy as np

# Set XLA flags before importing JAX
os.environ["XLA_FLAGS"] = " ".join([
    "--xla_force_host_platform_device_count=8",
    "--xla_dump_to=/tmp/xla",
    "--xla_dump_hlo_as_text",
    "--xla_dump_hlo_pass_re=spmd"
])

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from functools import partial


def test_axis_index_strategy():
    """Test if the axis_index strategy works for partition-id simulation"""
    
    # Create 1D mesh
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ('a',))
    
    @partial(jax.shard_map, mesh=mesh, in_specs=(P(),), out_specs=P('a'))
    def extract_partition_slice(full_array):
        # This mimics what the partition-id HLO does
        partition_id = jax.lax.axis_index('a')  # Should be 0,1,2,3 on different devices
        start_idx = partition_id * 2  # Each device gets 2 elements
        return jax.lax.dynamic_slice(full_array, (start_idx,), (2,))
    
    # Test input
    full_array = jnp.arange(8, dtype=jnp.float32)  # [0,1,2,3,4,5,6,7]
    
    # Run the function
    result = extract_partition_slice(full_array)
    
    print("Input (replicated):", full_array)
    print("Output (sharded):", result) 
    print("Output shape:", result.shape)
    
    # Expected: should be [0,1,2,3,4,5,6,7] distributed as 4 shards of 2 elements each
    # When assembled, should reconstruct the original array
    
    return jnp.allclose(result, full_array)


if __name__ == "__main__":
    is_identity = test_axis_index_strategy()
    print(f"Is identity: {is_identity}")