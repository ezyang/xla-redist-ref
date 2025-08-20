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
import difflib


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
                # Parse source-target pairs and use first mesh axis
                axis_name = self.device_mesh.axis_names[0]
                # For now, implement as identity - collective-permute needs more sophisticated handling
                # This would need to parse the pairs and implement the permutation logic
                return jax.lax.ppermute(operand, axis_name, perm=[(i, i) for i in range(self.device_mesh.devices.shape[0])])
            
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


def test_hlo_interpreter_and_extract_stablehlo(
    hlo_text: str,
    mesh: Mesh,
    in_specs: P,
    out_specs: P,
    array_shape: Tuple[int, ...]
) -> Tuple[bool, str, str]:
    """
    Test HLO interpreter and extract both StableHLO and post-SPMD HLO from the resulting JAX function.
    
    Returns (is_identity, stablehlo_text, post_spmd_hlo_text)
    """
    # Clear the dump directory to capture new HLO files
    dump_dir = "/tmp/xla"
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir, exist_ok=True)
    
    # Convert HLO to JAX function
    jax_func = hlo_to_jax_function(hlo_text, mesh)
    
    # Wrap in shard_map
    sharded_func = partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs
    )(jax_func)
    
    # Create test input as f32 to match original trace
    test_input = jnp.arange(np.prod(array_shape), dtype=jnp.float32).reshape(array_shape)
    test_input_sharded = jax.device_put(test_input, NamedSharding(mesh, in_specs))
    
    # JIT compile the JAX function and extract StableHLO
    jitted_func = jax.jit(sharded_func)
    lowered = jitted_func.lower(test_input_sharded)
    
    # Extract StableHLO representation
    stablehlo_text = lowered.as_text()
    
    # Compile to trigger HLO dumps
    exe = lowered.compile()
    
    # Find the post-SPMD HLO file
    hlo_files = glob.glob(os.path.join(dump_dir, "*after_spmd-partitioning*.txt"))
    if not hlo_files:
        post_spmd_hlo_text = "No post-SPMD HLO file found"
    else:
        # If multiple files, take the most recent one
        hlo_file = max(hlo_files, key=os.path.getmtime)
        
        # Read the post-SPMD HLO text from the dumped file
        with open(hlo_file, 'r') as f:
            post_spmd_hlo_text = f.read()
    
    # Run the function to test correctness
    result = jitted_func(test_input_sharded)
    
    # Check if result equals input (identity function)
    is_identity = jnp.allclose(result, test_input)
    
    return is_identity, stablehlo_text, post_spmd_hlo_text


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
    is_identity, _, _ = test_hlo_interpreter_and_extract_stablehlo(hlo_text, mesh, in_specs, out_specs, array_shape)
    return is_identity


def normalize_hlo_line(line: str) -> str:
    """
    Normalize an HLO line for comparison by:
    1. Removing variable name prefixes (%var.123 -> var)
    2. Normalizing whitespace
    3. Removing metadata and layout info
    """
    line = line.strip()
    
    # Remove metadata like {metadata={...}}
    line = re.sub(r',\s*metadata=\{[^}]*\}', '', line)
    
    # Remove layout info like {1,0}
    line = re.sub(r'\{[0-9,\s]*\}', '', line)
    
    # Normalize variable names: %param_0.123 -> param_0
    line = re.sub(r'%([^.\s,)]+)(?:\.[0-9]+)?', r'\1', line)
    
    # Normalize whitespace
    line = ' '.join(line.split())
    
    return line


def extract_hlo_operations(hlo_text: str) -> List[Dict[str, Any]]:
    """
    Extract operations from HLO text, normalizing for comparison.
    Returns list of operation dictionaries with normalized content.
    """
    operations = []
    
    for line in hlo_text.split('\n'):
        line = line.strip()
        
        # Skip structural lines
        if (not line or line.startswith('#') or line.startswith('//') or
            line.startswith('HloModule') or line.startswith('ENTRY') or 
            line.startswith('}') or line.startswith('{') or 
            'computation_layout' in line):
            continue
            
        # Check if this is an operation assignment
        if '=' in line and not line.startswith('ROOT'):
            # Extract variable name and normalize the operation
            var_match = re.match(r'%?([^=\s]+)\s*=\s*(.+)', line)
            if var_match:
                var_name = var_match.group(1)
                operation = var_match.group(2)
                
                # Extract operation type
                op_match = re.match(r'[^a-zA-Z]*([a-zA-Z][a-zA-Z0-9_-]*)', operation)
                op_type = op_match.group(1) if op_match else 'unknown'
                
                operations.append({
                    'var': var_name,
                    'operation': normalize_hlo_line(operation),
                    'op_type': op_type,
                    'original_line': line
                })
    
    return operations


def match_hlo_operations(ops1: List[Dict], ops2: List[Dict]) -> List[Tuple[int, int, float]]:
    """
    Match operations between two HLO operation lists based on similarity.
    Returns list of (index1, index2, similarity_score) tuples.
    """
    matches = []
    used_indices2 = set()
    
    for i, op1 in enumerate(ops1):
        best_match = None
        best_score = 0.0
        
        for j, op2 in enumerate(ops2):
            if j in used_indices2:
                continue
                
            # Calculate similarity score
            score = 0.0
            
            # Operation type match (high weight)
            if op1['op_type'] == op2['op_type']:
                score += 0.6
            
            # Operation text similarity (medium weight)
            op_similarity = difflib.SequenceMatcher(None, op1['operation'], op2['operation']).ratio()
            score += 0.4 * op_similarity
            
            if score > best_score and score > 0.3:  # Minimum threshold
                best_score = score
                best_match = j
        
        if best_match is not None:
            matches.append((i, best_match, best_score))
            used_indices2.add(best_match)
    
    return matches


def word_diff(text1: str, text2: str) -> str:
    """
    Generate a word-level diff between two texts.
    """
    words1 = text1.split()
    words2 = text2.split()
    
    diff = difflib.unified_diff(words1, words2, lineterm='', n=3)
    return ' '.join(diff)


def analyze_operation_dependencies(operations: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Analyze the dependency chain of operations to understand data flow.
    """
    op_info = {}
    
    for i, op in enumerate(operations):
        var_name = op['var']
        op_info[var_name] = {
            'index': i,
            'type': op['op_type'],
            'shape': op.get('shape', 'unknown'),
            'operation': op['operation'],
            'operands': []
        }
        
        # Extract operands
        if 'operand' in op:
            op_info[var_name]['operands'] = [op['operand']]
        elif 'operands' in op:
            op_info[var_name]['operands'] = op['operands']
    
    return op_info


def trace_operation_chain(op_info: Dict[str, Dict], start_var: str, max_depth: int = 10) -> List[str]:
    """
    Trace the operation chain backwards from a given variable.
    """
    chain = []
    current = start_var
    depth = 0
    
    while current in op_info and depth < max_depth:
        info = op_info[current]
        chain.append(f"{current}: {info['type']} -> {info['shape']}")
        
        # Follow the first operand
        if info['operands']:
            current = info['operands'][0]
            depth += 1
        else:
            break
    
    return chain



def compare_hlos_smart_diff(
    original_hlo: str,
    reconstructed_hlo: str,
    original_label: str = "Original HLO",
    reconstructed_label: str = "Reconstructed HLO"
) -> str:
    """
    Compare two HLO texts using smart matching and word-level diffing.
    
    This function:
    1. Extracts and normalizes operations from both HLOs
    2. Matches operations based on semantic similarity
    3. Shows word-level diffs for matched operations
    4. Highlights unmatched operations
    """
    result = []
    result.append(f"=== Smart HLO Comparison: {original_label} vs {reconstructed_label} ===\n")
    
    # Extract operations
    ops1 = extract_hlo_operations(original_hlo)
    ops2 = extract_hlo_operations(reconstructed_hlo)
    
    result.append(f"Operations in {original_label}: {len(ops1)}")
    result.append(f"Operations in {reconstructed_label}: {len(ops2)}\n")
    
    # Match operations
    matches = match_hlo_operations(ops1, ops2)
    
    result.append(f"Matched operations: {len(matches)}/{max(len(ops1), len(ops2))}\n")
    
    # Analyze dependencies for better debugging
    deps1 = analyze_operation_dependencies(ops1)
    deps2 = analyze_operation_dependencies(ops2)
    
    # Show matched operations with differences
    matched_indices1 = set()
    matched_indices2 = set()
    
    for i1, i2, score in matches:
        matched_indices1.add(i1)
        matched_indices2.add(i2)
        
        op1 = ops1[i1]
        op2 = ops2[i2]
        
        if op1['operation'] != op2['operation']:
            result.append(f"DIFF [{op1['op_type']}] (similarity: {score:.2f}):")
            result.append(f"  {original_label}: {op1['operation']}")
            result.append(f"  {reconstructed_label}: {op2['operation']}")
            
            # Word-level diff
            words1 = op1['operation'].split()
            words2 = op2['operation'].split()
            
            if len(words1) < 20 and len(words2) < 20:  # Only for short operations
                matcher = difflib.SequenceMatcher(None, words1, words2)
                word_diff_parts = []
                
                for tag, i1_start, i1_end, i2_start, i2_end in matcher.get_opcodes():
                    if tag == 'equal':
                        word_diff_parts.append(' '.join(words1[i1_start:i1_end]))
                    elif tag == 'delete':
                        word_diff_parts.append(f"[-{' '.join(words1[i1_start:i1_end])}-]")
                    elif tag == 'insert':
                        word_diff_parts.append(f"[+{' '.join(words2[i2_start:i2_end])}+]")
                    elif tag == 'replace':
                        word_diff_parts.append(f"[-{' '.join(words1[i1_start:i1_end])}-]")
                        word_diff_parts.append(f"[+{' '.join(words2[i2_start:i2_end])}+]")
                
                result.append(f"  Word diff: {' '.join(word_diff_parts)}")
            
            # For reshape/transpose operations, show dependency chains
            if op1['op_type'] in ['reshape', 'transpose', 'bitcast']:
                var1 = op1['var']
                var2 = op2['var']
                
                chain1 = trace_operation_chain(deps1, var1, max_depth=5)
                chain2 = trace_operation_chain(deps2, var2, max_depth=5)
                
                if chain1 or chain2:
                    result.append(f"  Dependency chain analysis:")
                    result.append(f"    {original_label}:")
                    for step in chain1:
                        result.append(f"      {step}")
                    result.append(f"    {reconstructed_label}:")
                    for step in chain2:
                        result.append(f"      {step}")
            
            result.append("")
    
    # Show unmatched operations from original
    unmatched1 = [i for i in range(len(ops1)) if i not in matched_indices1]
    if unmatched1:
        result.append(f"Operations only in {original_label}:")
        for i in unmatched1:
            result.append(f"  - [{ops1[i]['op_type']}] {ops1[i]['operation']}")
        result.append("")
    
    # Show unmatched operations from reconstructed
    unmatched2 = [i for i in range(len(ops2)) if i not in matched_indices2]
    if unmatched2:
        result.append(f"Operations only in {reconstructed_label}:")
        for i in unmatched2:
            result.append(f"  + [{ops2[i]['op_type']}] {ops2[i]['operation']}")
        result.append("")
    
    return '\n'.join(result)


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
    
    is_identity, stablehlo_text, post_spmd_hlo_text = test_hlo_interpreter_and_extract_stablehlo(
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
