import numpy as np
import jax
from jax.sharding import Mesh
from typing import Tuple


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