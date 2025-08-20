import os
import shutil
import glob
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from typing import Tuple, Any
from mesh_utils import create_device_mesh


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
        HLO text representation from after SPMD pass
    """
    # Clear the dump directory to avoid old files
    dump_dir = "/tmp/xla"
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir, exist_ok=True)
    
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
    
    # Lower and compile the function (this will trigger HLO dumps)
    lowered = jax.jit(sharded_identity).lower(dummy_input)
    exe = lowered.compile()
    
    # Find the HLO file generated after SPMD pass
    # Look for files with "after_spmd-partitioning" in the name
    hlo_files = glob.glob(os.path.join(dump_dir, "*after_spmd-partitioning*.txt"))
    if not hlo_files:
        raise RuntimeError(f"No HLO files found after SPMD partitioning in {dump_dir}")
    
    # If multiple files, take the most recent one
    hlo_file = max(hlo_files, key=os.path.getmtime)
    
    # Read the HLO text from the dumped file
    with open(hlo_file, 'r') as f:
        hlo_text = f.read()
    
    return hlo_text