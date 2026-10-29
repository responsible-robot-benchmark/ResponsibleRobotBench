#!/usr/bin/env bash
# Source this file from any working directory; all writes stay in this project.
RRB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export COPPELIASIM_ROOT="$RRB_ROOT/.runtime/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$RRB_ROOT/.runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export PYTHONPATH="$RRB_ROOT:$RRB_ROOT/responsible_robot_bench/envs/rrb_manipulation${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
