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

# Test when axis_index gets evaluated
def test_axis_index_timing():
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ('a',))
    
    print("=== Testing when axis_index gets evaluated ===")
    
    def make_function_with_axis_index():
        print("Creating function with axis_index...")
        
        def inner_func(x):
            print("Inside inner_func, calling axis_index...")
            partition_id = jax.lax.axis_index('a')
            print(f"Got partition_id: {partition_id}")
            return x
        
        return inner_func
    
    print("Step 1: Create function")
    func = make_function_with_axis_index()
    
    print("Step 2: Wrap in shard_map")
    try:
        sharded_func = partial(jax.shard_map, mesh=mesh, in_specs=P(), out_specs=P())(func)
        print("shard_map created successfully")
    except Exception as e:
        print(f"shard_map creation failed: {e}")
        return
    
    print("Step 3: Call the sharded function")
    test_input = jnp.array([1, 2, 3, 4])
    try:
        result = sharded_func(test_input)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Execution failed: {e}")


if __name__ == "__main__":
    test_axis_index_timing()