#!/bin/bash
# Batch script for running Heisenberg / AKLT spin chain simulations
# Usage: bash run_simulations.sh
# Toggle RUN_* flags to choose which simulations to run.

# ---- Configuration ----
NUM_CORES=4                    # max cores for numpy/scipy (BLAS threads)
PYTHON=python3                 # python command
BOUNDARY="PBC"                 # PBC or OBC

# Toggle which simulations to run (true/false)
RUN_S_HALF_HEISENBERG=true
RUN_S1_HEISENBERG=false
RUN_S1_AKLT=false

# Site ranges
S_HALF_HEISENBERG_MIN=2
S_HALF_HEISENBERG_MAX=16

S1_HEISENBERG_MIN=2
S1_HEISENBERG_MAX=10

S1_AKLT_MIN=2
S1_AKLT_MAX=10
# ---- End Configuration ----

export OMP_NUM_THREADS=$NUM_CORES
export MKL_NUM_THREADS=$NUM_CORES
export OPENBLAS_NUM_THREADS=$NUM_CORES

echo "=============================================="
echo "Spin chain batch simulation"
echo "Cores: $NUM_CORES, Boundary: $BOUNDARY"
echo "=============================================="

if [ "$RUN_S_HALF_HEISENBERG" = true ]; then
    echo ""
    echo "--- S=1/2 Heisenberg (N=$S_HALF_HEISENBERG_MIN..$S_HALF_HEISENBERG_MAX) ---"
    for N in $(seq $S_HALF_HEISENBERG_MIN $S_HALF_HEISENBERG_MAX); do
        echo ""
        echo "[S=1/2 Heisenberg] N=$N  $(date +%H:%M:%S)"
        $PYTHON main.py -N $N -S 0.5 -b $BOUNDARY -m Heisenberg
        if [ $? -ne 0 ]; then
            echo "ERROR: S=1/2 Heisenberg N=$N failed!"
        fi
    done
fi

if [ "$RUN_S1_HEISENBERG" = true ]; then
    echo ""
    echo "--- S=1 Heisenberg (N=$S1_HEISENBERG_MIN..$S1_HEISENBERG_MAX) ---"
    for N in $(seq $S1_HEISENBERG_MIN $S1_HEISENBERG_MAX); do
        echo ""
        echo "[S=1 Heisenberg] N=$N  $(date +%H:%M:%S)"
        $PYTHON main.py -N $N -S 1 -b $BOUNDARY -m Heisenberg
        if [ $? -ne 0 ]; then
            echo "ERROR: S=1 Heisenberg N=$N failed!"
        fi
    done
fi

if [ "$RUN_S1_AKLT" = true ]; then
    echo ""
    echo "--- S=1 AKLT (N=$S1_AKLT_MIN..$S1_AKLT_MAX) ---"
    for N in $(seq $S1_AKLT_MIN $S1_AKLT_MAX); do
        echo ""
        echo "[S=1 AKLT] N=$N  $(date +%H:%M:%S)"
        $PYTHON main.py -N $N -S 1 -b $BOUNDARY -m AKLT
        if [ $? -ne 0 ]; then
            echo "ERROR: S=1 AKLT N=$N failed!"
        fi
    done
fi

echo ""
echo "=============================================="
echo "All simulations finished at $(date +%H:%M:%S)"
echo "=============================================="
