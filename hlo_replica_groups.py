import re
import numpy as np
from jax.sharding import Mesh
from hlo_parsing import HLOParseError


def parse_replica_groups(replica_groups_str: str, device_mesh: Mesh) -> str:
    """Parse replica_groups supporting both explicit and iota formats.
    
    Formats:
    - Explicit: {{0,1},{2,3}} or {}
    - Iota: [2,2]<=[4] or [2,10]<=[4,5]T(1,0)
    
    Returns the axis name to use for the collective operation.
    """
    if not replica_groups_str:
        raise HLOParseError("Missing replica_groups")
    
    if replica_groups_str.startswith('{'):
        return _parse_explicit_replica_groups(replica_groups_str, device_mesh)
    elif replica_groups_str.startswith('['):
        return _parse_iota_replica_groups(replica_groups_str, device_mesh)
    else:
        raise HLOParseError(f"Invalid replica_groups format: {replica_groups_str}")


def _parse_explicit_replica_groups(replica_groups_str: str, device_mesh: Mesh) -> str:
    """Parse explicit replica_groups format like {{0,1},{2,3}}"""
    # For explicit format, we need to infer which mesh axis corresponds to the groups
    # This is a simplified approach - use first available mesh axis
    mesh_axes = device_mesh.axis_names
    return mesh_axes[0] if mesh_axes else 'default'


def _parse_iota_replica_groups(replica_groups_str: str, device_mesh: Mesh) -> str:
    """Parse iota replica_groups format using IotaTileAssignment algorithm"""
    # Parse format: [dims]<=[reshape_dims]T?(transpose_perm)?
    match = re.match(r'\[([^\]]+)\]<=\[([^\]]+)\](?:T\(([^)]+)\))?', replica_groups_str)
    if not match:
        raise HLOParseError(f"Invalid iota replica_groups format: {replica_groups_str}")
    
    dims = [int(x.strip()) for x in match.group(1).split(',')]
    reshape_dims = [int(x.strip()) for x in match.group(2).split(',')]
    transpose_perm = None
    if match.group(3):
        transpose_perm = [int(x.strip()) for x in match.group(3).split(',')]
    
    # Apply the IotaTileAssignment algorithm
    N = np.prod(reshape_dims)
    iota_array = np.arange(N)  # [0,1,2,...,N-1]
    
    # Reshape to reshape_dims
    reshaped = iota_array.reshape(reshape_dims)
    
    # Apply transpose if present
    if transpose_perm:
        reshaped = np.transpose(reshaped, transpose_perm)
    
    # Final reshape to dims
    groups = reshaped.reshape(dims)
    
    # Convert to list of lists for comparison
    group_lists = [list(groups[i]) for i in range(groups.shape[0])]
    
    # Now determine which mesh axis corresponds to these groups
    # Generate expected groups for each mesh axis and compare
    mesh_shape_list = list(device_mesh.shape.values())
    axis_name = None
    
    for axis_idx in range(len(mesh_shape_list)):
        # Generate expected groups for this axis using the same algorithm
        expected_groups = []
        
        # For axis_idx, generate groups by fixing other coordinates and varying this one
        other_dims = [mesh_shape_list[i] for i in range(len(mesh_shape_list)) if i != axis_idx]
        if len(other_dims) == 0:
            # Single dimension mesh
            group = list(range(mesh_shape_list[axis_idx]))
            expected_groups = [group]
        else:
            for other_coords in np.ndindex(*other_dims):
                group = []
                for axis_val in range(mesh_shape_list[axis_idx]):
                    # Insert axis_val at position axis_idx
                    full_coords = []
                    other_idx = 0
                    for i in range(len(mesh_shape_list)):
                        if i == axis_idx:
                            full_coords.append(axis_val)
                        else:
                            full_coords.append(other_coords[other_idx])
                            other_idx += 1
                    
                    # Convert to flat device ID (row-major)
                    flat_id = 0
                    for i, coord in enumerate(full_coords):
                        flat_id = flat_id * mesh_shape_list[i] + coord
                    group.append(flat_id)
                expected_groups.append(sorted(group))
        
        # Sort both for comparison
        expected_groups_sorted = [sorted(g) for g in sorted(expected_groups)]
        group_lists_sorted = [sorted(g) for g in sorted(group_lists)]
        
        if expected_groups_sorted == group_lists_sorted:
            axis_name = device_mesh.axis_names[axis_idx]
            break
    
    if axis_name is None:
        raise HLOParseError(f"Could not match replica groups {group_lists} to any mesh axis")
    
    return axis_name