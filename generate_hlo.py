import os
import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec as P

# Set XLA flags before importing JAX
os.environ["XLA_FLAGS"] = " ".join([
    "--xla_force_host_platform_device_count=8",
    "--xla_dump_to=/tmp/xla",
    "--xla_dump_hlo_as_text",
    "--xla_dump_hlo_pass_re=spmd"
])

# Import the refactored modules
from mesh_utils import generate_mesh_axes, create_device_mesh
from hlo_generation import generate_hlo_for_sharding_constraints
from hlo_parsing import HLOParseError, HLOParser
from hlo_interpretation import HLOInterpreter, hlo_to_jax_function
from hlo_analysis import (
    hlo_interpreter_and_extract_stablehlo,
    test_hlo_interpreter,
    compare_hlos_smart_diff
)




if __name__ == "__main__":
    # Example usage: Generate HLO and test interpreter
    mesh_shape = (2, 2)  # 2x2 mesh for 4 devices
    array_shape = (4, 4)  # Small array for testing
    
    # Example: Shard input along both axes, output identity mapping
    in_spec = P('a', 'b')  # Shard along both mesh axes
    out_spec = P(None, ('a', 'b'))  # Different output sharding to test collectives
    
    hlo = generate_hlo_for_sharding_constraints(in_spec, out_spec, mesh_shape, array_shape)
    print(hlo)
    
    devices = np.array(jax.devices()[:4]).reshape(2, 2)
    mesh = Mesh(devices, ('a', 'b'))
    
    is_identity, stablehlo_text, post_spmd_hlo_text = hlo_interpreter_and_extract_stablehlo(
        hlo, mesh, in_spec, out_spec, array_shape
    )
    print(f"HLO interpreter test: {'PASS' if is_identity else 'FAIL'}")
    
    print("\n=== Original post-SPMD HLO ===")
    print(hlo)
    
    print("\n=== Post-SPMD HLO from JAX interpreter ===")
    print(post_spmd_hlo_text)
    
    print("\n=== Smart diff: Original vs JAX Post-SPMD HLO ===")
    diff = compare_hlos_smart_diff(
        hlo, 
        post_spmd_hlo_text,
        "Original post-SPMD HLO",
        "JAX interpreter post-SPMD HLO"
    )
    print(diff)
