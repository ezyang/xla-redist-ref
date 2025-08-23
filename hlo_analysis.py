import re
import os
import shutil
import glob
import difflib
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import shard_map
from typing import Tuple, Any, Dict, List
from hlo_interpretation import hlo_to_jax_function


def hlo_interpreter_and_extract_stablehlo(
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
    
    # Convert HLO to JAX function with sharding context
    sharding_context = {
        'input_spec': in_specs,
        'output_spec': out_specs
    }
    jax_func = hlo_to_jax_function(hlo_text, mesh, sharding_context)
    
    # Wrap in shard_map
    sharded_func = partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False  # Allow replication without static inference
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
    
    # Debug: Print actual vs expected values when test fails
    if not is_identity:
        print(f"\n=== HLO INTERPRETER TEST FAILURE DEBUG ===")
        print(f"Expected (input): {test_input}")
        print(f"Actual (output):  {result}")
        print(f"Shape - Expected: {test_input.shape}, Actual: {result.shape}")
        print(f"Close elements: {jnp.isclose(result, test_input).sum()}/{test_input.size}")
        if test_input.size <= 64:  # Only show diff for small arrays
            print(f"Difference: {result - test_input}")
        print(f"Max absolute difference: {jnp.max(jnp.abs(result - test_input))}")
        print("=== END DEBUG ===\n")
    
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