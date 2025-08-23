import re
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from typing import Dict, Any, List, Callable
from hlo_parsing import HLOParseError, HLOParser
from hlo_replica_groups import parse_replica_groups


class HLOInterpreter:
    """Interprets parsed HLO operations as JAX LAX primitives."""
    
    def __init__(self, device_mesh: Mesh, sharding_context):
        self.variables = {}
        self.device_mesh = device_mesh
        self.sharding_context = sharding_context or {}
    
    def _translate_dtype(self, dtype: str) -> str:
        """Translate HLO dtype to JAX dtype."""
        return dtype.replace('s32', 'int32').replace('u32', 'uint32').replace('f32', 'float32')
    
        
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
                    
                    # Compute flattened index (row-major order): i0 * S1*S2*... + i1 * S2*S3*... + ...
                    partition_id = indices[0]
                    for i in range(1, len(indices)):
                        partition_id = partition_id * sizes[i] + indices[i]
                
                # Convert to target dtype and return as scalar
                return partition_id.astype(jnp.dtype(target_dtype))
            
            # Constant operation: uses parsed constant value
            case {'type': 'constant', 'constant_value': value}:
                return value
            
            # Fallback for constant operations without parsed value
            case {'type': 'constant', 'shape': shape, 'dtype': dtype}:
                return jnp.zeros(shape, dtype=jnp.dtype(self._translate_dtype(dtype)))
            
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
            case {'type': 'slice', 'operand': operand_name, 'slice': slice_ranges}:
                operand = self.variables[operand_name]
                
                # Apply slicing using parsed ranges
                result = operand
                for i, (start, end) in enumerate(slice_ranges):
                    result = jax.lax.slice_in_dim(result, start, end, axis=i)
                return result
            
            # Dynamic slice operation: requires 'operands' and 'dynamic_slice_sizes'
            case {'type': 'dynamic-slice', 'operands': operand_names, 'dynamic_slice_sizes': slice_sizes}:
                # First operand is the array to slice, rest are start indices
                array = self.variables[operand_names[0]]
                start_indices = [self.variables[name] for name in operand_names[1:]]
                
                # Use parsed slice sizes directly
                return jax.lax.dynamic_slice(array, start_indices, slice_sizes)
            
            # All-reduce operation: requires 'operand'
            case {'type': 'all-reduce', 'operand': operand_name, **kwargs}:
                operand = self.variables[operand_name]
                
                # First try to use replica_groups if available
                replica_groups = kwargs.get('replica_groups', '')
                if replica_groups:
                    axis_name = parse_replica_groups(replica_groups, self.device_mesh)
                    return jax.lax.psum(operand, axis_name=axis_name)
                else:
                    # For all-reduce, we sum across all devices using the device mesh
                    axis_names = self.device_mesh.axis_names
                    return jax.lax.psum(operand, axis_name=axis_names)
            
            # Reshape operation: requires 'operand' and 'shape'
            case {'type': 'reshape', 'operand': operand_name, 'shape': shape}:
                operand = self.variables[operand_name]
                return jax.lax.reshape(operand, shape)
            
            # All-to-all operation: requires 'operand' and 'dimensions'
            case {'type': 'all-to-all', 'operand': operand_name, 'dimensions': dimensions, **kwargs}:
                operand = self.variables[operand_name]
                split_axis = dimensions[0] if dimensions else 0
                concat_axis = split_axis
                
                # Parse replica_groups using unified function
                replica_groups = kwargs.get('replica_groups', '')
                axis_name = parse_replica_groups(replica_groups, self.device_mesh)
                
                return jax.lax.all_to_all(operand, axis_name, split_axis, concat_axis)
            
            # All-gather operation: requires 'operand' and 'dimensions' 
            case {'type': 'all-gather', 'operand': operand_name, 'dimensions': dimensions, **kwargs}:
                operand = self.variables[operand_name]
                
                # First try to use replica_groups if available
                replica_groups = kwargs.get('replica_groups', '')
                if replica_groups:
                    axis_name = parse_replica_groups(replica_groups, self.device_mesh)
                else:
                    # Fallback to sharding context
                    input_spec = self.sharding_context.get('input_spec', None)
                    output_spec = self.sharding_context.get('output_spec', None)
                    
                    if input_spec is not None and output_spec is not None:
                        # We have sharding context - use it to determine the axis
                        # For X -> P(None) transformations, gather along the axis that X specifies
                        
                        # PartitionSpec is a tuple-like object, access it directly
                        if hasattr(input_spec, '__len__') and len(input_spec) > 0:
                            sharded_axis_name = input_spec[0]  # Get the first axis name
                            if sharded_axis_name in self.device_mesh.axis_names:
                                axis_name = sharded_axis_name
                            else:
                                axis_name = self.device_mesh.axis_names[0]  # Fallback
                        else:
                            axis_name = self.device_mesh.axis_names[0]  # Default
                    else:
                        # No context available, use default
                        axis_name = self.device_mesh.axis_names[0]
                
                concat_axis = dimensions[0] if dimensions else 0
                # Use tiled=True to concatenate along existing axis instead of creating new axis
                return jax.lax.all_gather(operand, axis_name, axis=concat_axis, tiled=True)
            
            # Collective-permute operation: requires 'operand' and 'source_target_pairs'
            case {'type': 'collective-permute', 'operand': operand_name, 'source_target_pairs': perm_pairs}:
                operand = self.variables[operand_name]
                
                # For multi-axis mesh, we need to use all axis names for ppermute
                # JAX ppermute expects axis_name to be a tuple for multi-axis permutation
                axis_names = self.device_mesh.axis_names
                
                if len(axis_names) == 1:
                    # Single axis case
                    return jax.lax.ppermute(operand, axis_names[0], perm=perm_pairs)
                else:
                    # Multi-axis case: use all axes together
                    return jax.lax.ppermute(operand, axis_names, perm=perm_pairs)
            
            # Subtract operation: requires 'operands' (two operands)
            case {'type': 'subtract', 'operands': operand_names}:
                if len(operand_names) >= 2:
                    lhs = self.variables[operand_names[0]]
                    rhs = self.variables[operand_names[1]]
                    return jax.lax.sub(lhs, rhs)
                else:
                    raise HLOParseError(f"Subtract operation requires at least 2 operands, got {len(operand_names)}")
            
            # Compare operation: requires 'operands' and 'direction'
            case {'type': 'compare', 'operands': operand_names, 'direction': direction}:
                if len(operand_names) >= 2:
                    lhs = self.variables[operand_names[0]]
                    rhs = self.variables[operand_names[1]]
                    # Map HLO comparison directions to JAX lax comparison functions
                    compare_ops = {
                        'EQ': jax.lax.eq,
                        'NE': jax.lax.ne,
                        'LT': jax.lax.lt,
                        'LE': jax.lax.le,
                        'GT': jax.lax.gt,
                        'GE': jax.lax.ge,
                    }
                    if direction in compare_ops:
                        return compare_ops[direction](lhs, rhs)
                    else:
                        raise HLOParseError(f"Unsupported comparison direction: {direction}")
                else:
                    raise HLOParseError(f"Compare operation requires at least 2 operands, got {len(operand_names)}")
            
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


def hlo_to_jax_function(hlo_text: str, device_mesh: Mesh, sharding_context=None) -> Callable:
    """Convert HLO text to a JAX function."""
    parser = HLOParser()
    operations = parser.parse(hlo_text)
    
    interpreter = HLOInterpreter(device_mesh, sharding_context)
    
    def jax_function(*inputs):
        return interpreter.interpret(operations, inputs)
    
    return jax_function