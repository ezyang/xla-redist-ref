"""Replica group utilities for JAX-like device meshes.

This module provides:
- compute_replica_groups: given a device mesh, axis_name(s), and optional
  per-axis index partitions, compute explicit replica groups (list of device ids).
- infer_axes_and_partitions: given a device mesh and explicit replica_groups,
  infer which mesh axes are grouped and optionally recover per-axis index
  partitions if the grouping factors as a product of independent per-axis blocks.

Assumptions
- mesh_coords: dict[int, tuple[int, ...]] mapping device_id -> coordinates
  of length k (number of mesh dims)
- mesh_shape: tuple[int, ...] of length k
- axis_names: dict[int, str] mapping dim index -> axis name (unique names)

Notes
- axis_name can be a single string or a tuple/list of strings. They must match
  values in axis_names.
- axis_index_groups (forward) can be:
  * None: groups are full fibers along the selected axes.
  * For a single axis: list[list[int]] describing index blocks along that axis.
  * For multiple axes: dict keyed by dim index or axis name with list[list[int]]
    per axis; groups are the Cartesian product of blocks across the selected axes
    within each fiber of the complement dims.

This file includes a small test harness under __main__.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


MeshCoords = Dict[int, Tuple[int, ...]]
AxisNames = Dict[int, str]


def _dims_for_axis_name(axis_names: AxisNames, axis_name: Union[str, Sequence[str]]) -> List[int]:
    name_to_dim = {v: k for k, v in axis_names.items()}
    if isinstance(axis_name, (list, tuple)):
        dims = []
        for n in axis_name:
            if n not in name_to_dim:
                raise ValueError(f"Unknown axis name {n!r}")
            dims.append(name_to_dim[n])
        return sorted(dims)
    else:
        if axis_name not in name_to_dim:
            raise ValueError(f"Unknown axis name {axis_name!r}")
        return [name_to_dim[axis_name]]


def _normalize_partitions(
    axis_index_groups: Optional[Union[List[List[int]], Dict[Union[int, str], List[List[int]]]]],
    dims: List[int],
    axis_names: AxisNames,
    mesh_shape: Tuple[int, ...],
):
    """Normalize axis_index_groups to a dict keyed by dim index -> list[list[int]].

    Accepts:
    - None
    - list[list[int]] for single axis in dims
    - dict keyed by dim index or axis name -> list[list[int]] for all dims
    Validates coverage and disjointness for each provided axis.
    """
    if axis_index_groups is None:
        return None

    result: Dict[int, List[List[int]]] = {}
    if isinstance(axis_index_groups, list):
        if len(dims) != 1:
            raise ValueError("List form of axis_index_groups only valid for a single axis")
        m = dims[0]
        result[m] = axis_index_groups
    elif isinstance(axis_index_groups, dict):
        name_to_dim = {v: k for k, v in axis_names.items()}
        for key, blocks in axis_index_groups.items():
            if isinstance(key, int):
                m = key
            else:
                if key not in name_to_dim:
                    raise ValueError(f"Unknown axis name in axis_index_groups: {key!r}")
                m = name_to_dim[key]
            result[m] = blocks
    else:
        raise TypeError("axis_index_groups must be None, list[list[int]], or dict")

    # Validate coverage and disjointness for each axis in dims
    for m in dims:
        if m not in result:
            raise ValueError(f"Missing axis_index_groups for axis dim {m}")
        blocks = result[m]
        flat = [i for b in blocks for i in b]
        if sorted(flat) != list(range(mesh_shape[m])):
            raise ValueError(
                f"axis_index_groups for dim {m} must cover 0..{mesh_shape[m]-1} exactly; got {sorted(flat)}"
            )
        # disjointness check
        seen = set()
        for b in blocks:
            for i in b:
                if i in seen:
                    raise ValueError(f"axis_index_groups for dim {m} has overlapping index {i}")
                seen.add(i)

    return result


def _fiber_key(coords: Tuple[int, ...], dims: Iterable[int]) -> Tuple[int, ...]:
    """Return tuple of coordinates with given dims removed (i.e., comp(M))."""
    k = list(range(len(coords)))
    dims_set = set(dims)
    return tuple(coords[i] for i in k if i not in dims_set)


def compute_replica_groups(
    mesh_coords: MeshCoords,
    mesh_shape: Tuple[int, ...],
    axis_names: AxisNames,
    axis_name: Union[str, Sequence[str]],
    axis_index_groups: Optional[Union[List[List[int]], Dict[Union[int, str], List[List[int]]]]] = None,
) -> List[List[int]]:
    """Compute replica groups given mesh and axis selection.

    If axis_index_groups is None, groups are full fibers over the selected axes.
    If provided, groups are formed by Cartesian products of per-axis blocks within each fiber.
    """
    dims = _dims_for_axis_name(axis_names, axis_name)
    partitions = _normalize_partitions(axis_index_groups, dims, axis_names, mesh_shape)

    # Partition devices by fiber key over comp(M)
    devs = sorted(mesh_coords.keys())
    fibers: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for d in devs:
        key = _fiber_key(mesh_coords[d], dims)
        fibers[key].append(d)

    groups: List[List[int]] = []

    if partitions is None:
        # Each fiber is a single group of size prod(d_m)
        for key in sorted(fibers.keys()):
            g = sorted(fibers[key])
            # Basic sanity: group size equals product of dims in M
            expected = 1
            for m in dims:
                expected *= mesh_shape[m]
            if len(g) != expected:
                raise ValueError(
                    f"Fiber {key} has {len(g)} devices, expected {expected} for dims {dims}"
                )
            groups.append(g)
        return groups

    # With per-axis partitions, form product of blocks across M within each fiber
    # Precompute block lookup per axis
    blocks_by_axis: Dict[int, List[List[int]]] = {m: [sorted(b) for b in partitions[m]] for m in dims}

    # For deterministic ordering, sort blocks by their smallest element
    for m in dims:
        blocks_by_axis[m].sort(key=lambda b: (len(b), b))

    # Build a mapping from per-axis index -> block id for filtering
    block_id_by_idx: Dict[int, Dict[int, int]] = {}
    for m in dims:
        bid = {}
        for idx, block in enumerate(blocks_by_axis[m]):
            for i in block:
                bid[i] = idx
        block_id_by_idx[m] = bid

    # For each fiber and for each combination of blocks, collect devices
    for key in sorted(fibers.keys()):
        fiber_devs = fibers[key]
        # group devices by their block ids tuple across dims
        bucket: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for d in fiber_devs:
            coords = mesh_coords[d]
            block_ids = tuple(block_id_by_idx[m][coords[m]] for m in dims)
            bucket[block_ids].append(d)

        # Determine expected number of buckets = product of number of blocks per axis
        expected_buckets = 1
        for m in dims:
            expected_buckets *= len(blocks_by_axis[m])

        if len(bucket) != expected_buckets:
            # Allow empty buckets would imply invalid partition vs mesh
            # Gather some diagnostics
            raise ValueError(
                f"Fiber {key}: observed {len(bucket)} block combinations, expected {expected_buckets}"
            )

        for combo in sorted(bucket.keys()):
            groups.append(sorted(bucket[combo]))

    return groups


def infer_axes_and_partitions(
    mesh_coords: MeshCoords,
    mesh_shape: Tuple[int, ...],
    axis_names: AxisNames,
    replica_groups: List[List[int]],
) -> Tuple[List[str], Optional[Dict[int, List[List[int]]]]]:
    """Infer axes and per-axis partitions from explicit replica_groups.

    Returns (axes_names, axis_index_groups_by_dim or None).
    """
    k = len(mesh_shape)
    devs = sorted(mesh_coords.keys())
    if not replica_groups:
        raise ValueError("replica_groups must be non-empty")

    # Quick validation: groups should be non-empty and cover only known devices
    for g in replica_groups:
        if not g:
            raise ValueError("replica_groups contain an empty group")
        for d in g:
            if d not in mesh_coords:
                raise ValueError(f"Device id {d} in replica_groups not in mesh_coords")

    # Build all subsets of dims, increasing size
    all_dims = list(range(k))
    subsets: List[Tuple[int, ...]] = []
    for r in range(1, k + 1):
        subsets.extend(itertools.combinations(all_dims, r))

    # Precompute fiber keys for devices for each candidate M? We'll compute per M.

    def aligns_with_M(M: Tuple[int, ...]) -> bool:
        comp = [i for i in all_dims if i not in M]
        # Group alignment: within each group, coords on comp are identical
        for g in replica_groups:
            # reference key for first device
            d0 = g[0]
            ref_key = tuple(mesh_coords[d0][i] for i in comp)
            for d in g[1:]:
                if tuple(mesh_coords[d][i] for i in comp) != ref_key:
                    return False
        # Per-fiber partition checks
        # Build mapping key -> groups in that fiber
        groups_by_key: Dict[Tuple[int, ...], List[List[int]]] = defaultdict(list)
        for g in replica_groups:
            d0 = g[0]
            key = tuple(mesh_coords[d0][i] for i in comp)
            groups_by_key[key].append(g)

        # Enumerate all fiber keys present in mesh (not just groups)
        # Build set of keys from all devices
        all_keys = set()
        for d in devs:
            all_keys.add(tuple(mesh_coords[d][i] for i in comp))

        # Expected fiber size
        expected_fiber_size = 1
        for m in M:
            expected_fiber_size *= mesh_shape[m]

        # For coverage check on tuples
        expected_tuples = set(itertools.product(*[range(mesh_shape[m]) for m in M]))

        # For each key, check disjointness and coverage
        for key in all_keys:
            groups_in_fiber = groups_by_key.get(key, [])
            # Build union of devices and check disjointness
            seen = set()
            union = []
            for g in groups_in_fiber:
                for d in g:
                    if d in seen:
                        return False
                    seen.add(d)
                    union.append(d)

            if len(union) != expected_fiber_size:
                return False

            # The union must be exactly the devices of this fiber
            expected_devs = [d for d in devs if tuple(mesh_coords[d][i] for i in comp) == key]
            if set(union) != set(expected_devs):
                return False

            # Coverage of tuples of M-indices
            observed_tuples = set(
                tuple(mesh_coords[d][m] for m in M) for d in union
            )
            if observed_tuples != expected_tuples:
                return False

        return True

    found_M: Optional[Tuple[int, ...]] = None
    for M in subsets:
        if aligns_with_M(M):
            # Debug: print identified M during inference
            # print(f"infer: candidate M accepted: {M}")
            found_M = M
            break

    if found_M is None:
        raise ValueError("Could not find a set of mesh dims that explain replica_groups")

    axes = [axis_names[m] for m in sorted(found_M)]

    # Phase 2: attempt to factor per-axis partitions
    M = tuple(sorted(found_M))
    comp = [i for i in all_dims if i not in M]

    # Build groups_by_key again for M
    groups_by_key: Dict[Tuple[int, ...], List[List[int]]] = defaultdict(list)
    for g in replica_groups:
        d0 = g[0]
        key = tuple(mesh_coords[d0][i] for i in comp)
        groups_by_key[key].append(g)

    if not groups_by_key:
        return axes, None

    # Pick a canonical fiber key
    key0 = sorted(groups_by_key.keys())[0]
    G_key0 = groups_by_key[key0]

    # If there is only one group per fiber, that's a full fiber (no partitions)
    # No need to recover per-axis partitions
    if len(G_key0) == 1:
        return axes, None

    # Build per-axis projections for each group in the fiber
    Pm: Dict[int, List[Tuple[frozenset[int], int]]] = {m: [] for m in M}
    # Also collect the observed tuples per group to check factorization
    tuples_in_group: Dict[int, set] = {}
    for gi, g in enumerate(G_key0):
        tuples_in_group[gi] = set(
            tuple(mesh_coords[d][m] for m in M) for d in g if tuple(mesh_coords[d][i] for i in comp) == key0
        )
        for m in M:
            P = {mesh_coords[d][m] for d in g if tuple(mesh_coords[d][i] for i in comp) == key0}
            Pm[m].append((frozenset(P), gi))

    # For each axis, check pairwise disjointness and full coverage
    partitions_by_axis: Dict[int, List[frozenset[int]]] = {}
    for m in M:
        blocks = [b for (b, _gi) in Pm[m]]
        # Merge identical blocks if present (groups might share same projection across axes)
        unique_blocks = list(set(blocks))
        # Check disjointness
        for a, b in itertools.combinations(unique_blocks, 2):
            if a & b:
                return axes, None
        # Check union equals full range
        union = set().union(*unique_blocks) if unique_blocks else set()
        if union != set(range(mesh_shape[m])):
            return axes, None
        partitions_by_axis[m] = sorted(unique_blocks, key=lambda s: (len(s), sorted(s)))

    # Factorization: for every group in G_key0, its tuples must equal product of per-axis blocks matching its projections
    for gi, g in enumerate(G_key0):
        # For each axis m, find the block in partitions_by_axis[m] that contains the group's projection
        blocks_for_g: List[Iterable[int]] = []
        for m in M:
            proj = next(b for (b, _gi2) in Pm[m] if _gi2 == gi)
            # Identify the matching block from partitions (proj should equal that block)
            matching = None
            for b in partitions_by_axis[m]:
                if proj == b:
                    matching = b
                    break
            if matching is None:
                return axes, None
            blocks_for_g.append(sorted(matching))

        prod = set(itertools.product(*blocks_for_g))
        if tuples_in_group[gi] != prod:
            return axes, None

    # Consistency across other fiber keys (order may permute)
    canonical = {m: {frozenset(b) for b in partitions_by_axis[m]} for m in M}
    for key, G in groups_by_key.items():
        # skip the canonical key
        if key == key0:
            continue
        # Recompute partitions for this fiber
        seen_blocks: Dict[int, set] = {m: set() for m in M}
        for g in G:
            for m in M:
                P = frozenset(mesh_coords[d][m] for d in g if tuple(mesh_coords[d][i] for i in comp) == key)
                seen_blocks[m].add(P)
        for m in M:
            if {frozenset(b) for b in seen_blocks[m]} != canonical[m]:
                return axes, None

    # Build deterministic axis_index_groups dict keyed by dim index
    result: Dict[int, List[List[int]]] = {}
    for m in M:
        blocks = [sorted(list(b)) for b in partitions_by_axis[m]]
        blocks.sort()
        result[m] = blocks

    return axes, result


# ------------------------------- Tests ------------------------------------


def _mk_mesh_coords(mesh_shape: Tuple[int, ...]) -> MeshCoords:
    """Create a dense mesh_coords mapping with row-major device ids."""
    coords = {}
    did = 0
    for idx in itertools.product(*[range(n) for n in mesh_shape]):
        coords[did] = tuple(idx)
        did += 1
    return coords


def _axis_names_for_k(k: int) -> AxisNames:
    # Default axis names x,y,z,w,a,b,...
    default = ["x", "y", "z", "w", "a", "b", "c", "d"]
    names = {}
    for i in range(k):
        names[i] = default[i] if i < len(default) else f"dim{i}"
    return names


def _round_trip(mesh_shape, axis_name, axis_index_groups=None):
    mesh_coords = _mk_mesh_coords(mesh_shape)
    axis_names = _axis_names_for_k(len(mesh_shape))
    groups = compute_replica_groups(mesh_coords, mesh_shape, axis_names, axis_name, axis_index_groups)
    axes, parts = infer_axes_and_partitions(mesh_coords, mesh_shape, axis_names, groups)
    return groups, axes, parts


def run_tests():
    tests = []

    # 1) Pure 2D fiber (full cross)
    tests.append((
        (2, 3), ("x", "y"), None,
    ))

    # 2) Blocked along x, full along y
    tests.append((
        (4, 2), ("x", "y"), {"x": [[0, 1], [2, 3]], "y": [[0], [1]]},
    ))

    # 3) Single axis with custom blocks
    tests.append((
        (4, 2), "x", [[0, 2], [1, 3]],
    ))

    # 4) 3D: group over x,z; full fibers
    tests.append((
        (2, 3, 2), ("x", "z"), None,
    ))

    # 5) 2D non-factorable within fiber (diagonal) -> expect None partitions
    # We build replica_groups directly by forward pass via per-axis blocks that do NOT factor.
    # We'll generate explicit diagonal groups manually for a 2x2 along (x,y)
    mesh_shape = (2, 2)
    mesh_coords = _mk_mesh_coords(mesh_shape)
    axis_names = _axis_names_for_k(2)
    # Build diagonal groups per fiber over comp(M)=() so it's global
    G_diag = []
    # devices at coords: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 under our id scheme
    # diag1: (0,0) and (1,1) ; diag2: (0,1) and (1,0)
    diag1 = [d for d, c in mesh_coords.items() if c in [(0, 0), (1, 1)]]
    diag2 = [d for d, c in mesh_coords.items() if c in [(0, 1), (1, 0)]]
    G_diag = [sorted(diag1), sorted(diag2)]

    def test_non_factorable():
        axes, parts = infer_axes_and_partitions(mesh_coords, mesh_shape, axis_names, G_diag)
        assert axes == ["x", "y"], f"Expected axes ['x','y'], got {axes}"
        assert parts is None, f"Expected None partitions for non-factorable grouping, got {parts}"

    # Execute tests
    for mesh_shape, axis_name, axis_index_groups in tests:
        groups, axes, parts = _round_trip(mesh_shape, axis_name, axis_index_groups)
        # Expected axes should be tuple(axis_name) if multi else [axis_name]
        want_axes = list(axis_name) if isinstance(axis_name, (list, tuple)) else [axis_name]
        assert axes == want_axes, f"axes mismatch: want {want_axes}, got {axes}"
        # When forward partitions None, inverse partitions should be None
        if axis_index_groups is None:
            assert parts is None, f"Expected None partitions, got {parts}"
        else:
            # Normalize to dict by dim to compare, only for single-axis list form or dict per-axis
            axis_names = _axis_names_for_k(len(mesh_shape))
            dims = _dims_for_axis_name(axis_names, axis_name)
            norm = _normalize_partitions(axis_index_groups, dims, axis_names, mesh_shape)
            if norm is None:
                assert parts is None
            else:
                # parts keys should be dims; compare sets of blocks per axis
                for m in dims:
                    want = sorted(sorted(b) for b in norm[m])
                    got = sorted(sorted(b) for b in (parts.get(m) or []))
                    assert want == got, f"Partitions mismatch on dim {m}: want {want}, got {got}"

    test_non_factorable()

    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
