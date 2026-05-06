#!/bin/bash
#SBATCH --job-name=retarget-oakink2
#SBATCH --output=logs/slurm_outputs/retarget-%A_%a.out
#SBATCH --error=logs/slurm_outputs/retarget-%A_%a.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a16

# Per-trajectory mano2dexhand fitting on OakInk2 indices, distributed across
# a SLURM job array. Each task picks one '<5hex>@<stage>' from the index file
# (line SLURM_ARRAY_TASK_ID + 1, 1-based) and runs the upstream retargeter.
#
# Outputs land at: data/retargeting/OakInk-v2/mano2<dexhand>/<full_basename>@<stage>.pkl
# (path baked in mano2dexhand.py:500-501).
#
# Usage:
#   N=$(wc -l < indices_cup.txt)
#   sbatch --array=0-$((N-1)) submit_retarget_oakink2_array.sh indices_cup.txt
#   # optional 2nd/3rd args:
#   sbatch --array=0-$((N-1)) submit_retarget_oakink2_array.sh indices_cup.txt inspire 2000
#
# To submit all three objects in one shot, generate a combined file first:
#   python filter_oakink2_indices.py --combined indices_combined.txt
#   N=$(wc -l < indices_combined.txt)
#   sbatch --array=0-$((N-1))%32 submit_retarget_oakink2_array.sh indices_combined.txt
# (the %32 limits concurrent running array tasks to 32 — adjust per cluster
# fairness.)

set -eo pipefail

INDEX_FILE="${1:?usage: $0 <indices.txt> [dexhand=inspire] [iter=2000]}"
DEXHAND="${2:-inspire}"
ITER="${3:-2000}"

if [ ! -f "$INDEX_FILE" ]; then
    echo "Index file '$INDEX_FILE' not found." >&2
    exit 1
fi
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Must be submitted as a job array (sbatch --array=0-N-1 ...)." >&2
    exit 1
fi

LINENO_=$((SLURM_ARRAY_TASK_ID + 1))
DATA_IDX=$(sed -n "${LINENO_}p" "$INDEX_FILE")
if [ -z "$DATA_IDX" ]; then
    echo "No index at line ${LINENO_} of ${INDEX_FILE} (file has $(wc -l < "$INDEX_FILE") lines)" >&2
    exit 1
fi
echo "[task $SLURM_ARRAY_TASK_ID] data_idx=${DATA_IDX}  dexhand=${DEXHAND}  iter=${ITER}"

cd ~/code/humanoid/ManipTrans
mkdir -p logs/slurm_outputs

module load conda/latest
conda activate maniptrans

# Same compiler-var unsets as other submit scripts in this repo (maniptrans
# conda env injects build vars that conflict with system gcc when extensions
# rebuild on first import).
unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CPP
unset CMAKE_ARGS CMAKE_PREFIX_PATH CONDA_BUILD_SYSROOT
unset AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP READELF
unset GCC GXX GCC_AR GCC_NM GCC_RANLIB
unset CC_FOR_BUILD CXX_FOR_BUILD NVCC_PREPEND_FLAGS
unset DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXX=/usr/bin/g++
export SETUPTOOLS_USE_DISTUTILS=stdlib

nvidia-smi -L

# Skip if the output already exists (safe to resume a partial array). The
# loader sorts listdir of anno_preview and picks the first match, so we
# replicate that here. Anchor the grep to '++seq__<HASH5>' to avoid false
# matches on substrings elsewhere in the filename. The dump dir uses
# `str(dexhand)` which appends `_rh` for the right-hand class
# (e.g. InspireRH.__str__() == 'inspire_rh').
DUMP_DIR="data/retargeting/OakInk-v2/mano2${DEXHAND}_rh"
HASH5="${DATA_IDX%@*}"
STAGE="${DATA_IDX#*@}"
EXPECTED=$(ls "data/OakInk-v2/anno_preview/" | sort | grep "++seq__${HASH5}" | head -n1 | sed "s/\.pkl\$//")
if [ -n "$EXPECTED" ] && [ -f "${DUMP_DIR}/${EXPECTED}@${STAGE}.pkl" ]; then
    echo "[task $SLURM_ARRAY_TASK_ID] already retargeted: ${DUMP_DIR}/${EXPECTED}@${STAGE}.pkl — skipping"
    exit 0
fi

python main/dataset/mano2dexhand.py \
    --data_idx "$DATA_IDX" \
    --side right \
    --dexhand "$DEXHAND" \
    --headless \
    --iter "$ITER"
