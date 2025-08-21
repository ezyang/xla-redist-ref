import re
import jax
import jax.lax
import jax.numpy as jnp
from jax.sharding import Mesh
from typing import Dict, Any, List, Callable
from hlo_parsing import HLOParseError, HLOParser


class HLOInterpreter:
    """Interprets parsed HLO operations as JAX LAX primitives."""
    
    def __init__(self, device_mesh: Mesh):
        self.variables = {}
        self.device_mesh = device_mesh
        
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
            
            # Partition-id operation: returns the current device's partition ID
            case {'type': 'partition-id', 'shape': shape, 'dtype': dtype}:
                # Compute partition ID using axis_index within JAX function execution context
                # This ensures each device gets the correct partition ID when executed
                target_dtype = dtype.replace('u32', 'uint32').replace('s32', 'int32').replace('f32', 'float32')
                
                if len(self.device_mesh.shape) == 1:
                    # Simple 1D case: partition ID is just the axis index
                    axis_name = self.device_mesh.axis_names[0]
                    partition_id = jax.lax.axis_index(axis_name)
                else:
                    # Multi-dimensional case: compute flattened index using axis_index and axis_size
                    # For 2D mesh (a,b), device at position (i,j) has flattened ID: i * size_b + j
                    axis_names = self.device_mesh.axis_names
                    
                    # Get axis indices and sizes
                    indices = [jax.lax.axis_index(name) for name in axis_names]
                    sizes = [jax.lax.axis_size(name) for name in axis_names]
                    
                    # Debug: Print what we're computing
                    debug_info = jnp.stack([indices[0], indices[1] if len(indices) > 1 else 0])
                    jax.debug.print("partition-id debug: axis indices = {}, sizes = {}", 
                                    debug_info, jnp.array(sizes))
                    
                    # Compute flattened index (row-major order): i0 * S1*S2*... + i1 * S2*S3*... + ...
                    partition_id = indices[0]
                    for i in range(1, len(indices)):
                        partition_id = partition_id * sizes[i] + indices[i]
                    
                    jax.debug.print("partition-id: computed partition_id = {}", partition_id)
                
                # Convert to target dtype and return as scalar
                return partition_id.astype(jnp.dtype(target_dtype))
            
            # Constant operation: requires 'operands' containing the constant value
            case {'type': 'constant', 'operands': operands, 'shape': shape, 'dtype': dtype}:
                # Parse constant value from operands list
                if operands and isinstance(operands[0], str):
                    # Handle cases like ['{0', '2', '4', '6}'] - join and parse
                    if operands[0].startswith('{') and len(operands) > 1:
                        # Reconstruct the full constant string
                        const_str = ''.join(operands)
                        # Remove braces and parse as array
                        values_str = const_str.strip('{}')
                        if values_str:
                            if dtype.startswith('s') or dtype.startswith('u'):  # integer types
                                values = [int(x.strip()) for x in values_str.split(',') if x.strip()]
                            else:  # float types
                                values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
                            return jnp.array(values, dtype=jnp.dtype(dtype.replace('s32', 'int32').replace('u32', 'uint32').replace('f32', 'float32')))
                        else:
                            # Empty constant
                            return jnp.array([], dtype=jnp.dtype(dtype.replace('s32', 'int32').replace('u32', 'uint32').replace('f32', 'float32')))
                    else:
                        # Single value
                        if dtype.startswith('s') or dtype.startswith('u'):  # integer types
                            value = int(operands[0])
                        else:  # float types  
                            value = float(operands[0])
                        return jnp.array(value, dtype=jnp.dtype(dtype.replace('s32', 'int32').replace('u32', 'uint32').replace('f32', 'float32')))
                else:
                    # Default to zeros if no operands
                    return jnp.zeros(shape, dtype=jnp.dtype(dtype.replace('s32', 'int32').replace('u32', 'uint32').replace('f32', 'float32')))
            
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
            
            # Dynamic slice operation: requires 'operands' and 'dynamic_slice_sizes'
            case {'type': 'dynamic-slice', 'operands': operand_names, 'dynamic_slice_sizes': slice_sizes}:
                # First operand is the array to slice, rest are start indices
                array = self.variables[operand_names[0]]
                start_indices = [self.variables[name] for name in operand_names[1:]]
                
                # Convert slice_sizes to list if it's a single value
                if isinstance(slice_sizes, (int, list)):
                    sizes = slice_sizes if isinstance(slice_sizes, list) else [slice_sizes]
                else:
                    sizes = [slice_sizes]
                
                # Debug: print what we're slicing
                jax.debug.print("dynamic-slice: array[0] from {} = {}, start_indices = {}, sizes = {}", 
                               operand_names[0], array, start_indices, sizes)
                
                # Use JAX dynamic_slice
                result = jax.lax.dynamic_slice(array, start_indices, sizes)
                jax.debug.print("dynamic-slice result: {}", result)
                return result
            
            # All-reduce operation: requires 'operand'
            case {'type': 'all-reduce', 'operand': operand_name}:
                operand = self.variables[operand_name]
                # For all-reduce, we sum across all devices using the device mesh
                axis_names = self.device_mesh.axis_names
                return jax.lax.psum(operand, axis_name=axis_names)
            
            # Reshape operation: requires 'operand' and 'shape'
            case {'type': 'reshape', 'operand': operand_name, 'shape': shape}:
                operand = self.variables[operand_name]
                return jax.lax.reshape(operand, shape)
            
            # All-to-all operation: requires 'operand' and 'dimensions'
            case {'type': 'all-to-all', 'operand': operand_name, 'dimensions': dimensions}:
                operand = self.variables[operand_name]
                # Use the specified dimension for all-to-all communication
                split_axis = dimensions[0] if dimensions else 0
                concat_axis = split_axis
                axis_name = self.device_mesh.axis_names[0]  # Use first mesh axis
                return jax.lax.all_to_all(operand, axis_name, split_axis, concat_axis)
            
            # All-gather operation: requires 'operand' and 'dimensions' 
            case {'type': 'all-gather', 'operand': operand_name, 'dimensions': dimensions, **kwargs}:
                operand = self.variables[operand_name]
                # Use the specified dimension for all-gather communication
                axis_name = self.device_mesh.axis_names[0]  # Use first mesh axis
                concat_axis = dimensions[0] if dimensions else 0
                # Use tiled=True to concatenate along existing axis instead of creating new axis
                return jax.lax.all_gather(operand, axis_name, axis=concat_axis, tiled=True)
            
            # Collective-permute operation: requires 'operand' and 'source_target_pairs'
            case {'type': 'collective-permute', 'operand': operand_name, 'source_target_pairs': pairs}:
                operand = self.variables[operand_name]
                
                # Parse source-target pairs from string like "{{0,0},{2,1},{1,2},{3,3}}"
                perm_pairs = []
                # Extract pairs using regex: find all {number,number} patterns
                import re
                pair_matches = re.findall(r'\{(\d+),(\d+)\}', pairs)
                for src, tgt in pair_matches:
                    perm_pairs.append((int(src), int(tgt)))
                
                # For multi-axis mesh, we need to use all axis names for ppermute
                # JAX ppermute expects axis_name to be a tuple for multi-axis permutation
                axis_names = self.device_mesh.axis_names
                
                if len(axis_names) == 1:
                    # Single axis case
                    return jax.lax.ppermute(operand, axis_names[0], perm=perm_pairs)
                else:
                    # Multi-axis case: use all axes together
                    return jax.lax.ppermute(operand, axis_names, perm=perm_pairs)
            
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


def hlo_to_jax_function(hlo_text: str, device_mesh: Mesh) -> Callable:
    """Convert HLO text to a JAX function."""
    parser = HLOParser()
    operations = parser.parse(hlo_text)
    
    interpreter = HLOInterpreter(device_mesh)
    
    def jax_function(*inputs):
        return interpreter.interpret(operations, inputs)
    
    return jax_function