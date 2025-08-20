import os
os.environ["XLA_FLAGS"] = " ".join([
    "--xla_force_host_platform_device_count=8",
    "--xla_dump_to=/tmp/xla_dumps",
    "--xla_dump_hlo_as_text",
    "--xla_dump_include_timestamp",
    "--xla_dump_hlo_pass_re=SPMD|Spmd|spmd|ShardingPropagation"
])

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import shard_map
from typing import Tuple, Any, Dict, List
import re


def generate_mesh_axes(mesh_shape: Tuple[int, ...]) -> Tuple[str, ...]:
    """Generate sequential axis names 'a', 'b', 'c', ... for mesh dimensions."""
    return tuple(chr(ord('a') + i) for i in range(len(mesh_shape)))


def create_device_mesh(mesh_shape: Tuple[int, ...]) -> Mesh:
    """Create a device mesh with the specified shape and sequential axis names."""
    world_size = 8
    total_devices = np.prod(mesh_shape)
    
    if total_devices != world_size:
        raise ValueError(f"Mesh shape {mesh_shape} requires {total_devices} devices, but world size is {world_size}")
    
    devices = np.array(jax.devices("cpu")[:world_size]).reshape(mesh_shape)
    axes = generate_mesh_axes(mesh_shape)
    
    return Mesh(devices, axes)


def generate_hlo_for_sharding_constraints(
    in_specs: Any,
    out_specs: Any, 
    mesh_shape: Tuple[int, ...],
    array_shape: Tuple[int, ...]
) -> str:
    """
    Generate HLO for an identity function with input/output sharding constraints.
    
    Args:
        in_specs: Input sharding specification (PartitionSpec or similar)
        out_specs: Output sharding specification (PartitionSpec or similar) 
        mesh_shape: Shape of the device mesh, e.g., (4, 2) for 4x2 mesh
        array_shape: Shape of the array to process
        
    Returns:
        HLO text representation
    """
    mesh = create_device_mesh(mesh_shape)
    
    def sharded_identity(x):
        # Apply input sharding constraint
        x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, in_specs))
        
        # Identity operation (keep value alive)
        y = x + 0
        
        # Apply output sharding constraint
        y = jax.lax.with_sharding_constraint(y, NamedSharding(mesh, out_specs))
        
        return y
    
    # Create a dummy input array
    dummy_input = jnp.ones(array_shape)
    
    # Lower and compile the function
    lowered = jax.jit(sharded_identity).lower(dummy_input)
    exe = lowered.compile()
    
    # Extract HLO after SPMD partitioning (post-compilation)
    # This will include the collectives inserted by SPMD
    hlo_text = exe.as_text()
    
    return hlo_text


if __name__ == "__main__":
    # Example usage
    mesh_shape = (4, 2)  # 4x2 mesh
    array_shape = (1024, 1024)
    
    # Example 1: Shard along first mesh axis for input, replicate for output
    in_spec = P('a', None)
    out_spec = P(None, None)
    
    hlo = generate_hlo_for_sharding_constraints(in_spec, out_spec, mesh_shape, array_shape)
    print("=== HLO for sharding example ===")
    print(hlo[:1000] + "..." if len(hlo) > 1000 else hlo)