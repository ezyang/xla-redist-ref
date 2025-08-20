import re
import jax
import jax.lax
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