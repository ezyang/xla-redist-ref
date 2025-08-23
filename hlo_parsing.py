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
            shape = tuple(int(d.strip()) for d in dims_str.split(','))
        
        return shape, dtype
    
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
            elif attr_name == 'replica_groups' and attr_string[i] == '[':
                # Handle replica_groups=[2,2]<=[4] syntax
                start = i
                bracket_count = 0
                while i < len(attr_string):
                    if attr_string[i] == '[':
                        bracket_count += 1
                    elif attr_string[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            # Look for optional <=[N] part
                            i += 1
                            if i < len(attr_string) and attr_string[i:i+2] == '<=':
                                # Skip past <=[N] part
                                while i < len(attr_string) and attr_string[i] not in ', \t':
                                    i += 1
                            break
                    i += 1
                attr_value = attr_string[start:i].strip()
            else:
                # Handle simple values (no braces)
                start = i
                while i < len(attr_string) and attr_string[i] not in ', \t':
                    i += 1
                attr_value = attr_string[start:i].strip()
            
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
            elif attr_name == 'source_target_pairs':
                # Keep source_target_pairs as string for collective-permute parsing
                attributes[attr_name] = attr_value
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