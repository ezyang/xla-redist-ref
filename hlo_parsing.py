import re
from typing import Tuple, Dict, Any, List, Optional


class HLOParseError(Exception):
    """Raised when HLO parsing encounters an unknown or unsupported operation."""
    pass


class HLOParser:
    """Generic HLO parser that operates line by line."""
    
    def __init__(self):
        self.variables = {}  # Maps variable names to their JAX equivalents
        self.operations = []  # List of parsed operations
        
    def parse_shape(self, shape_str: str) -> Tuple[Tuple[int, ...], str]:
        """Parse HLO shape string like 'f32[1024,1024]' or 'u32[]' -> ((1024, 1024), 'f32') or ((), 'u32')"""
        match = re.match(r'(\w+)\[([^\]]*)\]', shape_str)
        if not match:
            raise HLOParseError(f"Cannot parse shape: {shape_str}")
        
        dtype = match.group(1)
        dims_str = match.group(2)
        
        if dims_str.strip() == "":
            shape = ()  # Scalar shape
        else:
            dims_list = self.parse_comma_separated_integers(dims_str)
            shape = tuple(dims_list)
        
        return shape, dtype
    
    def parse_comma_separated_integers(self, value_str: str) -> List[int]:
        """Parse comma-separated integers from string like '2,8' into [2, 8]"""
        if isinstance(value_str, int):
            return [value_str]
        elif isinstance(value_str, list):
            return value_str
        elif isinstance(value_str, str):
            if not value_str.strip():
                return []
            return [int(x.strip()) for x in value_str.split(',') if x.strip()]
        else:
            return [value_str]

    def parse_regex_integer_pairs(self, input_str: str, pattern: str) -> List[Tuple[int, int]]:
        """Generic method to parse integer pairs using regex pattern"""
        pairs = []
        for match in re.finditer(pattern, input_str):
            first, second = int(match.group(1)), int(match.group(2))
            pairs.append((first, second))
        return pairs

    def parse_slice_attribute(self, slice_spec: str) -> List[Tuple[int, int]]:
        """Parse slice specification like '[0:2], [1:3]' into list of (start, end) tuples"""
        return self.parse_regex_integer_pairs(slice_spec, r'\[(\d+):(\d+)\]')

    def parse_source_target_pairs_attribute(self, pairs_str: str) -> List[Tuple[int, int]]:
        """Parse source-target pairs from string like '{{0,0},{2,1},{1,2},{3,3}}'"""
        return self.parse_regex_integer_pairs(pairs_str, r'\{(\d+),(\d+)\}')

    def parse_dynamic_slice_sizes_attribute(self, sizes_str: str) -> List[int]:
        """Parse dynamic slice sizes from string like '2,8' into [2, 8]"""
        return self.parse_comma_separated_integers(sizes_str)

    def parse_attributes(self, attr_string: str) -> Dict[str, Any]:
        """Parse HLO attributes like 'dimensions={2}', 'channel_id=1', or 'slice={[0:2], [0:1]}'"""
        attributes = {}
        
        # Handle nested braces by finding balanced braces for complex attributes
        i = 0
        while i < len(attr_string):
            # Look for attribute name
            name_match = re.match(r'(\w+)=', attr_string[i:])
            if not name_match:
                # Skip to next character if no match
                i += 1
                continue
            
            attr_name = name_match.group(1)
            i += name_match.end()
            
            if i >= len(attr_string):
                break
                
            # Parse the value after the '='
            if attr_string[i] == '{':
                # Handle braced values with potential nesting
                brace_count = 0
                start = i
                while i < len(attr_string):
                    if attr_string[i] == '{':
                        brace_count += 1
                    elif attr_string[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            i += 1
                            break
                    i += 1
                
                attr_value = attr_string[start:i]
                # Remove outer braces
                if attr_value.startswith('{') and attr_value.endswith('}'):
                    attr_value = attr_value[1:-1].strip()
            elif attr_name == 'replica_groups':
                # Handle both explicit {{0,1},{2,3}} and iota [2,2]<=[4]T(1,0) formats
                attr_value = self._parse_replica_groups_value(attr_string[i:])
                i += len(attr_value)
            else:
                # Handle simple values (no braces)
                start = i
                while i < len(attr_string) and attr_string[i] not in ', \t':
                    i += 1
                attr_value = attr_string[start:i].strip()
            
            # Parse different attribute types
            if attr_name == 'dimensions':
                # Parse dimensions as list of integers
                attributes[attr_name] = self.parse_comma_separated_integers(attr_value)
            elif attr_name == 'slice':
                # Parse slice specification into structured format
                attributes[attr_name] = self.parse_slice_attribute(attr_value)
            elif attr_name == 'source_target_pairs':
                # Parse source_target_pairs into structured format
                attributes[attr_name] = self.parse_source_target_pairs_attribute(attr_value)
            elif attr_name == 'dynamic_slice_sizes':
                # Parse dynamic slice sizes into structured format
                attributes[attr_name] = self.parse_dynamic_slice_sizes_attribute(attr_value)
            elif attr_name.endswith('_id') or (attr_value.isdigit() if attr_value else False):
                # Parse numeric attributes like channel_id
                attributes[attr_name] = int(attr_value)
            else:
                # Store other attributes as strings
                attributes[attr_name] = attr_value
            
            # Skip whitespace and commas
            while i < len(attr_string) and attr_string[i] in ', \t':
                i += 1
                
        return attributes
    
    def parse_constant_operand(self, operand_str: str, dtype: str) -> Any:
        """Parse constant operand value based on its dtype"""
        import jax.numpy as jnp
        
        if operand_str.startswith('{') and operand_str.endswith('}'):
            # Handle braced constant like '{0, 0, 1, 1}'
            values_str = operand_str.strip('{}')
            if values_str:
                if dtype.startswith('s') or dtype.startswith('u'):  # integer types
                    values = [int(x.strip()) for x in values_str.split(',') if x.strip()]
                else:  # float types
                    values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
                return jnp.array(values, dtype=jnp.dtype(self._translate_dtype(dtype)))
            else:
                # Empty constant
                return jnp.array([], dtype=jnp.dtype(self._translate_dtype(dtype)))
        else:
            # Single value without braces
            if dtype.startswith('s') or dtype.startswith('u'):  # integer types
                value = int(operand_str)
            else:  # float types  
                value = float(operand_str)
            return jnp.array(value, dtype=jnp.dtype(self._translate_dtype(dtype)))

    def _translate_dtype(self, dtype: str) -> str:
        """Translate HLO dtype to JAX dtype."""
        return dtype.replace('s32', 'int32').replace('u32', 'uint32').replace('f32', 'float32')

    def parse_operands(self, operand_string: str) -> List[str]:
        """Parse operand list like '%param_0.3, %param_1' -> ['param_0.3', 'param_1']
        Also handles constant values like '{0, 0, 1, 1}' as a single operand."""
        operands = []
        i = 0
        while i < len(operand_string):
            # Skip whitespace
            while i < len(operand_string) and operand_string[i] in ' \t':
                i += 1
            
            if i >= len(operand_string):
                break
            
            # Check if we have a braced expression
            if operand_string[i] == '{':
                # Find matching closing brace
                brace_count = 0
                start = i
                while i < len(operand_string):
                    if operand_string[i] == '{':
                        brace_count += 1
                    elif operand_string[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            i += 1
                            break
                    i += 1
                operands.append(operand_string[start:i])
            else:
                # Regular operand - find next comma or end
                start = i
                while i < len(operand_string) and operand_string[i] != ',':
                    i += 1
                operand = operand_string[start:i].strip().lstrip('%')
                if operand:
                    operands.append(operand)
            
            # Skip comma if present
            if i < len(operand_string) and operand_string[i] == ',':
                i += 1
        
        return operands

    def _parse_replica_groups_value(self, value_str: str) -> str:
        """Parse replica_groups attribute value supporting both explicit and iota formats.
        
        Grammar:
        - explicit: {{0,1},{2,3}} or {} or {{},{}}  
        - iota: [2,2]<=[4] or [2,10]<=[4,5]T(1,0)
        """
        value_str = value_str.strip()
        
        if value_str.startswith('{'):
            # Explicit format: parse nested braces
            return self._parse_explicit_replica_groups(value_str)
        elif value_str.startswith('['):
            # Iota format: parse [dims]<=[reshape]T?(perm)?
            return self._parse_iota_replica_groups(value_str)
        else:
            raise HLOParseError(f"Invalid replica_groups format: {value_str}")
    
    def _parse_explicit_replica_groups(self, value_str: str) -> str:
        """Parse explicit replica_groups format like {{0,1},{2,3}} or {}"""
        brace_count = 0
        i = 0
        while i < len(value_str):
            if value_str[i] == '{':
                brace_count += 1
            elif value_str[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return value_str[:i+1]
            i += 1
        raise HLOParseError(f"Unmatched braces in replica_groups: {value_str}")
    
    def _parse_iota_replica_groups(self, value_str: str) -> str:
        """Parse iota replica_groups format like [2,2]<=[4] or [2,10]<=[4,5]T(1,0)"""
        # Use regex to match the full iota pattern
        pattern = r'\[[\d,]+\]<=\[[\d,]+\](?:T\([\d,]+\))?'
        match = re.match(pattern, value_str)
        if not match:
            raise HLOParseError(f"Invalid iota replica_groups format: {value_str}")
        return match.group(0)

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
            
            # Special handling for constant operations - parse the value
            if op_name == 'constant' and operands and dtype:
                result['constant_value'] = self.parse_constant_operand(operands[0], dtype)
            elif len(operands) == 1:
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