# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alpamayo-R1 is a Reasoning Vision-Language-Action (VLA) model for autonomous driving that integrates a Vision-Language Model (Qwen3-VL) with a diffusion-based action generation pipeline. It produces both reasoning traces (Chain-of-Thought) and vehicle trajectories.

## Development Commands

```bash
# Environment setup
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active

# Run inference test
python src/alpamayo_r1/test_inference.py

# Format code before committing
pre-commit format
```

## Architecture

### Core Pipeline Flow
```
Image + Text → VLM (Qwen3-VL) → Reasoning Trace → Expert Model + Diffusion → Trajectory
```

### Key Components

**AlpamayoR1** (`models/alpamayo_r1.py`) - Main model orchestrating three sub-components:
- VLM Backbone: Processes images/text to generate reasoning traces
- Expert Model: Transformer decoder refining action predictions using VLM context
- Diffusion Sampler: Generates trajectory samples via iterative denoising

**Action Space** (`action_space/`) - Trajectory ↔ action conversions:
- `UnicycleAccelCurvatureActionSpace`: Kinematic bicycle model with acceleration and curvature controls
- 64 waypoints per trajectory, 0.1s time steps
- Normalized bounds: acceleration ±9.8 m/s², curvature ±0.2 rad/m

**Diffusion** (`diffusion/`) - Flow matching implementation:
- `FlowMatching`: Euler integration from Gaussian noise to action space
- 10 denoising steps by default, guided by expert model via `step_fn` callback

**Models** (`models/`) - Supporting components:
- `ReasoningVLA`: Base class managing VLM integration and trajectory tokenization
- `PerWaypointActionInProjV2`: Projects noisy actions to expert embeddings using Fourier encoding
- `delta_tokenizer.py`: Discrete trajectory tokenizer for encoding trajectories as LLM tokens

**Geometry** (`geometry/`) - Rotation and coordinate transformations (SO3→yaw, Euler conversions)

### Token Management
The model injects trajectory information into the LLM via special tokens:
- Discrete trajectory tokens: `<i0>` to `<i767>`
- Trajectory delimiters: `<|traj_history_start|>`, `<|traj_future_start|>`, etc.

### Configuration
Uses Hydra for configuration. `AlpamayoR1Config` (in `config.py`) extends `ReasoningVLAConfig` with diffusion, action space, and expert model configs.

## Code Conventions

- Line length: 100 characters (Ruff)
- Requires Python 3.12
- Commits require sign-off (`git commit -s`)
- Follow existing conventions in relevant modules
