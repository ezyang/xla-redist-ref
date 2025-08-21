# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an XLA redistribution reference implementation that focuses on HLO (High-Level Operations) generation, parsing, interpretation, and analysis for JAX distributed computing. The codebase provides tools to generate HLO from JAX sharding constraints, parse HLO text representations, interpret HLO operations as JAX primitives, and perform sophisticated analysis and comparison of HLO code.

## Development Setup

Use `uv` for dependency management:

```bash
# Setup environment (run codex_setup.sh or manually):
uv sync --frozen
source .venv/bin/activate
```

## Running the Code

The main entry point demonstrates the full pipeline:

```bash
python generate_hlo.py
```

This will:
1. Generate HLO from JAX sharding constraints
2. Parse and interpret the HLO
3. Test the interpreter against the original JAX function
4. Perform HLO comparison analysis

## Code Architecture

The codebase follows a modular architecture with clear separation of concerns:

### Core Modules

- **`mesh_utils.py`** - Device mesh creation and axis name generation utilities
- **`hlo_generation.py`** - Generates HLO text from JAX operations with sharding constraints
- **`hlo_parsing.py`** - Contains `HLOParser` class that parses HLO text into structured operation dictionaries
- **`hlo_interpretation.py`** - Contains `HLOInterpreter` class that executes parsed HLO operations as JAX primitives
- **`hlo_analysis.py`** - Testing, comparison, and analysis utilities for HLO code
- **`generate_hlo.py`** - Main orchestration script that imports and demonstrates all modules

### Key Classes and Flow

1. **HLO Generation Flow**: `generate_hlo_for_sharding_constraints()` creates JAX functions with sharding constraints, compiles them, and extracts the resulting HLO from XLA dumps in `/tmp/xla`

2. **HLO Parsing Flow**: `HLOParser` uses regex-based line-by-line parsing to convert HLO text into structured dictionaries with operation types, operands, shapes, and attributes

3. **HLO Interpretation Flow**: `HLOInterpreter` uses pattern matching to map parsed HLO operations to corresponding JAX LAX primitives, enabling execution of HLO as JAX functions

4. **Analysis Flow**: Functions in `hlo_analysis.py` provide testing (identity function verification), HLO comparison with smart matching, and dependency analysis

### XLA Configuration

The code sets specific XLA flags before JAX import:
- Forces 8 host platform devices for multi-device simulation
- Dumps HLO to `/tmp/xla` directory
- Enables text-based HLO dumps for the SPMD partitioning pass

### Distributed Computing Concepts

The codebase works with JAX's distributed computing primitives:
- **Device Meshes**: Multi-dimensional arrays of devices with named axes ('a', 'b', 'c', ...)
- **Sharding Constraints**: `PartitionSpec` objects that specify how arrays are distributed across device mesh axes
- **Collective Operations**: All-reduce, all-to-all, collective-permute operations for inter-device communication

### Testing and Validation

The interpreter validation works by:
1. Converting HLO back to JAX functions
2. Running both original and reconstructed functions on identical inputs
3. Verifying that results match (identity function test)
4. Providing detailed debugging output on mismatches

### HLO Analysis Features

- **Smart Diff**: Matches operations between HLO variants based on semantic similarity rather than exact text matching
- **Dependency Analysis**: Traces operation chains to understand data flow
- **Word-level Diffing**: Shows precise differences in operation parameters
- **Operation Categorization**: Groups operations by type for structured comparison