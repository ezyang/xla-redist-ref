import os
import shutil

# Set XLA flags before importing JAX
os.environ["XLA_FLAGS"] = " ".join([
    "--xla_force_host_platform_device_count=8",
    "--xla_dump_to=/tmp/xla",
    "--xla_dump_hlo_as_text",
    "--xla_dump_hlo_pass_re=spmd"
])

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import shard_map
from typing import Tuple, Any, Dict, List, Callable, Optional
import re
from functools import partial
import glob


def generate_mesh_axes(mesh_shape: Tuple[int, ...]) -> Tuple[str, ...]:
    """Generate sequential axis names 'a', 'b', 'c', ... for mesh dimensions."""
    return tuple(chr(ord('a') + i) for i in range(len(mesh_shape)))


def create_device_mesh(mesh_shape: Tuple[int, ...]) -> Mesh:
    """Create a device mesh with the specified shape and sequential axis names."""
    total_devices = np.prod(mesh_shape)
    available_devices = len(jax.devices("cpu"))
    
    if total_devices > available_devices:
        raise ValueError(f"Mesh shape {mesh_shape} requires {total_devices} devices, but only {available_devices} available")
    
    devices = np.array(jax.devices("cpu")[:total_devices]).reshape(mesh_shape)
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


class HLOParseError(Exception):
    """Raised when HLO parsing encounters an unknown or unsupported operation."""
    pass


class HLOParser:
    """Generic HLO parser that operates line by line."""
    
    def __init__(self):
        self.variables = {}  # Maps variable names to their JAX equivalents
        self.operations = []  # List of parsed operations
        
    def parse_shape(self, shape_str: str) -> Tuple[Tuple[int, ...], str]:
        """Parse HLO shape string like 'f32[1024,1024]' -> ((1024, 1024), 'f32')"""
        match = re.match(r'(\w+)\[([^\]]+)\]', shape_str)
        if not match:
            raise HLOParseError(f"Cannot parse shape: {shape_str}")
        
        dtype = match.group(1)
        dims_str = match.group(2)
        
        if dims_str.strip() == "":
            shape = ()
        else:
            shape = tuple(int(d.strip()) for d in dims_str.split(','))
        
        return shape, dtype
    
    def parse_attributes(self, attr_string: str) -> Dict[str, Any]:
        """Parse HLO attributes like 'dimensions={2}', 'channel_id=1', or 'slice={[0:2], [0:1]}'"""
        attributes = {}
        
        # Find all attribute=value pairs (both {value} and plain value)
        attr_pattern = r'(\w+)=(\{[^}]*\}|[^,\s]+)'
        for match in re.finditer(attr_pattern, attr_string):
            attr_name = match.group(1)
            attr_value = match.group(2).strip()
            
            # Handle braced values
            if attr_value.startswith('{') and attr_value.endswith('}'):
                attr_value = attr_value[1:-1].strip()
            
            # Parse different attribute types
            if attr_name == 'dimensions':
                # Parse dimensions as list of integers
                if attr_value:
                    attributes[attr_name] = [int(d.strip()) for d in attr_value.split(',')]
                else:
                    attributes[attr_name] = []
            elif attr_name == 'slice':
                # Keep slice specification as string for later parsing
                attributes[attr_name] = attr_value
            elif attr_name.endswith('_id') or attr_value.isdigit():
                # Parse numeric attributes like channel_id
                attributes[attr_name] = int(attr_value)
            else:
                # Store other attributes as strings
                attributes[attr_name] = attr_value
                
        return attributes
    
    def parse_operands(self, operand_string: str) -> List[str]:
        """Parse operand list like '%param_0.3, %param_1' -> ['param_0.3', 'param_1']"""
        operands = []
        for operand in operand_string.split(','):
            operand = operand.strip().lstrip('%')
            if operand:
                operands.append(operand)
        return operands

    def parse_operation(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single HLO operation line generically."""
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#') or line.startswith('//'):
            return None
            
        # Skip metadata lines and structural elements
        if (line.startswith('HloModule') or line.startswith('ENTRY') or 
            line.startswith('}') or line.startswith('ROOT') or 
            line.startswith('%fused_computation') or line.startswith('{') or
            'computation_layout' in line or 'allow_spmd' in line):
            return None
            
        # Parse variable assignment: %var = shape op(...) [attributes] [metadata]
        assign_match = re.match(r'%([^=\s]+)\s*=\s*(.+)', line)
        if not assign_match:
            return None
            
        var_name = assign_match.group(1)
        rest_of_line = assign_match.group(2).strip()
        
        # Extract shape at beginning: f32[2,2]{1,0} or tuple shape (f32[...], f32[...])
        tuple_shape_match = re.match(r'\(([^)]+)\)\s+(.+)', rest_of_line)
        if tuple_shape_match:
            # Handle tuple return types
            tuple_elements = tuple_shape_match.group(1)
            operation_part = tuple_shape_match.group(2)
            shape, dtype = 'tuple', 'tuple'  # Mark as tuple type
        else:
            shape_match = re.match(r'(\w+\[[^\]]*\](?:\{[^}]*\})?)\s+(.+)', rest_of_line)
            if shape_match:
                shape_str = shape_match.group(1)
                # Remove layout info {1,0} from shape
                clean_shape_str = re.sub(r'\{[^}]*\}', '', shape_str)
                shape, dtype = self.parse_shape(clean_shape_str)
                operation_part = shape_match.group(2)
            else:
                shape, dtype = None, None
                operation_part = rest_of_line
        
        # Parse operation name and operands with attributes
        # Pattern: operation_name(operands), attr1={value1}, attr2={value2}, metadata={...}
        main_match = re.match(r'(\w+(?:-\w+)*)\(([^)]*)\)(.*)$', operation_part)
        if not main_match:
            raise HLOParseError(f"Cannot parse operation format: {line}")
        
        op_name = main_match.group(1)
        operands_str = main_match.group(2)
        attributes_str = main_match.group(3)
        
        # Build result dictionary
        result = {
            'type': op_name,
            'var': var_name,
            'shape': shape,
            'dtype': dtype
        }
        
        # Parse operands
        if operands_str.strip():
            operands = self.parse_operands(operands_str)
            if len(operands) == 1:
                result['operand'] = operands[0]
            else:
                result['operands'] = operands
        
        # Parse attributes
        if attributes_str.strip():
            attributes = self.parse_attributes(attributes_str)
            result.update(attributes)
        
        return result
    
    def parse(self, hlo_text: str) -> List[Dict[str, Any]]:
        """Parse HLO text and return list of operations."""
        self.operations = []
        self.variables = {}
        
        for line_num, line in enumerate(hlo_text.split('\n'), 1):
            try:
                op = self.parse_operation(line)
                if op:
                    self.operations.append(op)
            except HLOParseError as e:
                raise HLOParseError(f"Line {line_num}: {e}")
        
        return self.operations


class HLOInterpreter:
    """Interprets parsed HLO operations as JAX LAX primitives."""
    
    def __init__(self):
        self.variables = {}
        
    def interpret_operation(self, op: Dict[str, Any], inputs: List[Any]) -> Any:
        """Interpret a single HLO operation as JAX LAX primitives using pattern matching."""
        
        match op:
            # Parameter operation: requires 'operand' field with parameter index
            case {'type': 'parameter', 'operand': param_idx}:
                param_index = int(param_idx)
                if param_index >= len(inputs):
                    raise ValueError(f"Parameter index {param_index} out of range")
                return inputs[param_index]
            
            # Add operation: requires 'operands' field with list of operand names
            case {'type': 'add', 'operands': operand_names}:
                operands = [self.variables[name] for name in operand_names]
                if len(operands) < 2:
                    raise HLOParseError(f"Add operation requires at least 2 operands: {op}")
                return jax.lax.add(operands[0], operands[1])
            
            # Copy operation: requires 'operand' field
            case {'type': 'copy', 'operand': operand_name}:
                return self.variables[operand_name]  # Identity operation
            
            # Concatenate operation: requires 'operands' and 'dimensions'
            case {'type': 'concatenate', 'operands': operand_names, 'dimensions': dims}:
                operands = [self.variables[name] for name in operand_names]
                axis = dims[0]  # Take first dimension
                return jax.lax.concatenate(operands, axis)
            
            # Transpose operation: requires 'operand' and 'dimensions'
            case {'type': 'transpose', 'operand': operand_name, 'dimensions': permutation}:
                operand = self.variables[operand_name]
                return jax.lax.transpose(operand, permutation)
            
            # Bitcast operation: requires 'operand' and 'shape'
            case {'type': 'bitcast', 'operand': operand_name, 'shape': new_shape}:
                operand = self.variables[operand_name]
                return jax.lax.reshape(operand, new_shape)
            
            # Slice operation: requires 'operand' and 'slice'
            case {'type': 'slice', 'operand': operand_name, 'slice': slice_spec}:
                operand = self.variables[operand_name]
                
                # Extract slice ranges
                ranges = []
                for match in re.finditer(r'\[(\d+):(\d+)\]', slice_spec):
                    start, end = int(match.group(1)), int(match.group(2))
                    ranges.append(slice(start, end))
                
                # Apply slicing
                result = operand
                for i, s in enumerate(ranges):
                    result = jax.lax.slice_in_dim(result, s.start, s.stop, axis=i)
                return result
            
            # All-reduce operation: requires 'operand'
            case {'type': 'all-reduce', 'operand': operand_name}:
                operand = self.variables[operand_name]
                # For all-reduce, we sum across all devices
                return jax.lax.psum(operand, axis_name=None)
            
            # Reshape operation: requires 'operand' and 'shape'
            case {'type': 'reshape', 'operand': operand_name, 'shape': shape}:
                operand = self.variables[operand_name]
                return jax.lax.reshape(operand, shape)
            
            # All-to-all operation: requires 'operand' and 'dimensions'
            case {'type': 'all-to-all', 'operand': operand_name, 'dimensions': dimensions}:
                operand = self.variables[operand_name]
                raise NotImplementedError()
            
            # Collective-permute operation: requires 'operand' and 'source_target_pairs'
            case {'type': 'collective-permute', 'operand': operand_name, 'source_target_pairs': pairs}:
                operand = self.variables[operand_name]
                raise NotImplementedError()
            
            # Catch-all for missing required fields
            case {'type': op_type}:
                raise HLOParseError(f"Operation '{op_type}' missing required fields or not implemented: {op}")
            
            # Catch-all for malformed operations
            case _:
                raise HLOParseError(f"Malformed operation (missing 'type' field): {op}")
    
    def interpret(self, operations: List[Dict[str, Any]], inputs: List[Any]) -> Any:
        """Interpret a sequence of HLO operations."""
        self.variables = {}
        
        result = None
        for op in operations:
            value = self.interpret_operation(op, inputs)
            self.variables[op['var']] = value
            result = value  # Last operation is the result
            
        return result


def hlo_to_jax_function(hlo_text: str) -> Callable:
    """Convert HLO text to a JAX function."""
    parser = HLOParser()
    operations = parser.parse(hlo_text)
    
    interpreter = HLOInterpreter()
    
    def jax_function(*inputs):
        return interpreter.interpret(operations, inputs)
    
    return jax_function


def test_hlo_interpreter_and_extract_stablehlo(
    hlo_text: str,
    mesh: Mesh,
    in_specs: P,
    out_specs: P,
    array_shape: Tuple[int, ...]
) -> Tuple[bool, str]:
    """
    Test HLO interpreter and extract StableHLO from the resulting JAX function.
    
    Returns (is_identity, stablehlo_text)
    """
    # Convert HLO to JAX function
    jax_func = hlo_to_jax_function(hlo_text)
    
    # Wrap in shard_map
    sharded_func = partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs
    )(jax_func)
    
    # Create test input
    test_input = jnp.arange(np.prod(array_shape)).reshape(array_shape)
    test_input_sharded = jax.device_put(test_input, NamedSharding(mesh, in_specs))
    
    # JIT compile the JAX function and extract StableHLO
    jitted_func = jax.jit(sharded_func)
    lowered = jitted_func.lower(test_input_sharded)
    
    # Extract StableHLO representation
    stablehlo_text = lowered.as_text()
    
    # Run the function to test correctness
    result = jitted_func(test_input_sharded)
    
    # Check if result equals input (identity function)
    is_identity = jnp.allclose(result, test_input)
    
    return is_identity, stablehlo_text


def test_hlo_interpreter(
    hlo_text: str,
    mesh: Mesh,
    in_specs: P,
    out_specs: P,
    array_shape: Tuple[int, ...]
) -> bool:
    """
    Test if the HLO interpreter correctly reproduces the original function behavior.
    
    Returns True if the function acts as identity on the full semantic array.
    """
    is_identity, _ = test_hlo_interpreter_and_extract_stablehlo(hlo_text, mesh, in_specs, out_specs, array_shape)
    return is_identity


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
    
    is_identity, stablehlo_text = test_hlo_interpreter_and_extract_stablehlo(
        hlo, mesh, in_spec, out_spec, array_shape
    )
    print(f"HLO interpreter test: {'PASS' if is_identity else 'FAIL'}")
    print("\n=== StableHLO from JAX interpreter ===")
    print(stablehlo_text)
