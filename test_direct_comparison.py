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
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial

from hlo_generation import generate_hlo_for_sharding_constraints
from hlo_interpretation import hlo_to_jax_function


def test_direct_vs_interpreter():
    """Compare direct JAX implementation vs our interpreter for P(None) -> P('a')"""
    
    # Setup
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices, ('a',))
    test_input = jnp.arange(8, dtype=jnp.float32)
    
    print("=== Testing P(None) -> P('a') transformation ===")
    print(f"Input: {test_input}")
    print()
    
    # Method 1: Direct JAX implementation (what SHOULD happen)
    @partial(jax.shard_map, mesh=mesh, in_specs=(P(),), out_specs=P('a'))
    def direct_transform(x):
        partition_id = jax.lax.axis_index('a')
        start_idx = partition_id * 2  # Each device gets 2 elements
        return jax.lax.dynamic_slice(x, (start_idx,), (2,))
    
    direct_result = direct_transform(test_input)
    print(f"Direct JAX result: {direct_result}")
    print(f"Direct JAX shape: {direct_result.shape}")
    print(f"Direct is identity: {jnp.allclose(direct_result, test_input)}")
    print()
    
    # Method 2: Our interpreter approach
    hlo = generate_hlo_for_sharding_constraints(P(None), P('a'), (4,), (8,))
    jax_func = hlo_to_jax_function(hlo, mesh)
    
    interpreter_func = partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=P(None), 
        out_specs=P('a'),
        check_vma=False
    )(jax_func)
    
    # Create properly sharded input for interpreter
    test_input_sharded = jax.device_put(test_input, NamedSharding(mesh, P(None)))
    interpreter_result = interpreter_func(test_input_sharded)
    
    print(f"Interpreter result: {interpreter_result}")
    print(f"Interpreter shape: {interpreter_result.shape}")
    print(f"Interpreter is identity: {jnp.allclose(interpreter_result, test_input)}")
    print()
    
    # Compare results
    print(f"Results match: {jnp.allclose(direct_result, interpreter_result)}")
    
    # The key insight: both should produce the SAME sharded result
    # The "identity" test might be wrong - P(None)->P('a') changes the sharding!
    

if __name__ == "__main__":
    test_direct_vs_interpreter()