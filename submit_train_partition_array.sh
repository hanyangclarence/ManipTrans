#!/bin/bash
#SBATCH --job-name=oakink2-stage2-part
#SBATCH --output=logs/slurm_outputs/stage2-part-%A_%a.out
#SBATCH --error=logs/slurm_outputs/stage2-part-%A_%a.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a16

# Stage-2 (residual manipulation) trained as a SLURM array, with the dataset
# indices partitioned into fixed-size chunks. Each array task k trains one
# policy on chunk k (lines [k*n+1 : (k+1)*n] of indices_<object>.txt).
#
# Cap: at most MAX_CHUNKS=20 policies per (object, n) — extra chunks are
# silently dropped (matches the user's "subsample and skip" plan).
#
# Usage (compute chunks, then sbatch with the right array range):
#   N=5; OBJ=cup
#   M=$(wc -l < indices_$OBJ.txt)
#   CHUNKS=$(( ((M + N - 1) / N) < 20 ? ((M + N - 1) / N) : 20 ))
#   sbatch --array=0-$((CHUNKS-1))%8 submit_train_partition_array.sh $OBJ $N
#
# n="all" trains a single policy on every trajectory in the file:
#   sbatch --array=0-0 submit_train_partition_array.sh stick all
#
# Experiment dir name embeds n + zero-padded chunk index + the array job id,
# so eval_partition_results.sh can glob all chunks of a submission together:
#   oakink2_inspire_<obj>_n<N>_chunk<kk>_arr<jobid>
#
# Each policy runs for the SLURM time limit (6h) — no max_epochs / early_stop.

set -eo pipefail

OBJECT="${1:?usage: $0 <cup|spoon|stick> <n|all>}"
N_ARG="${2:?usage: $0 <cup|spoon|stick> <n|all>}"

case "$OBJECT" in
    cup|spoon|stick) ;;
    *) echo "Unsupported object '$OBJECT'." >&2; exit 1 ;;
esac

cd ~/code/humanoid/ManipTrans
mkdir -p logs/slurm_outputs

INDEX_FILE="indices_${OBJECT}.txt"
if [ ! -f "$INDEX_FILE" ]; then
    echo "Index file '$INDEX_FILE' not found. Run filter_oakink2_indices.py first." >&2
    exit 1
fi
M=$(wc -l < "$INDEX_FILE")

MAX_CHUNKS=20
if [ "$N_ARG" = "all" ]; then
    N_PER=$M
    TOTAL_CHUNKS=1
elif [[ "$N_ARG" =~ ^[1-9][0-9]*$ ]]; then
    N_PER=$N_ARG
    TOTAL_CHUNKS=$(( (M + N_PER - 1) / N_PER ))
else
    echo "n must be 'all' or a positive integer, got '$N_ARG'." >&2
    exit 1
fi
CHUNKS=$(( TOTAL_CHUNKS < MAX_CHUNKS ? TOTAL_CHUNKS : MAX_CHUNKS ))

# Sanity: each array task ID must map to a valid chunk.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "$CHUNKS" ]; then
    echo "[task $TASK_ID] beyond CHUNKS=$CHUNKS (n=$N_ARG, M=$M); exiting." >&2
    exit 0
fi

START=$(( TASK_ID * N_PER + 1 ))
END=$(( (TASK_ID + 1) * N_PER ))
if [ "$END" -gt "$M" ]; then END=$M; fi
INDICES=$(sed -n "${START},${END}p" "$INDEX_FILE" | paste -sd ,)

if [ -z "$INDICES" ]; then
    echo "[task $TASK_ID] empty chunk slice (start=$START, end=$END)" >&2
    exit 1
fi

CHUNK_PADDED=$(printf '%02d' "$TASK_ID")
ARR_ID="${SLURM_ARRAY_JOB_ID:-local$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT="oakink2_inspire_${OBJECT}_n${N_ARG}_chunk${CHUNK_PADDED}_arr${ARR_ID}"

# Skip-if-already-trained: if any prior array left a "best" checkpoint at
# runs/<expt-prefix>__*/nn/<expt-prefix>.pth, treat this chunk as done and
# exit 0. The chunk identity is (object, n_arg, chunk_index) — the array
# job id is allowed to differ, so re-running the same sbatch picks up only
# the missing chunks. Set FORCE_RETRAIN=1 in the env to override.
EXPT_PREFIX="oakink2_inspire_${OBJECT}_n${N_ARG}_chunk${CHUNK_PADDED}_arr"
if [ -z "${FORCE_RETRAIN:-}" ]; then
    EXISTING=$(ls runs/${EXPT_PREFIX}*/nn/${EXPT_PREFIX}*.pth 2>/dev/null | head -1 || true)
    if [ -n "$EXISTING" ]; then
        echo "[task $TASK_ID] chunk already trained; skipping. Found: $EXISTING"
        echo "  (set FORCE_RETRAIN=1 to retrain anyway.)"
        exit 0
    fi
fi

echo "Object:      $OBJECT"
echo "N per chunk: $N_PER  (n_arg=$N_ARG)"
echo "Chunks:      $CHUNKS  (of $TOTAL_CHUNKS total, capped at $MAX_CHUNKS)"
echo "Task:        $TASK_ID  -> lines $START..$END  ($(echo $INDICES | tr ',' '\n' | wc -l) trajectories)"
echo "Experiment:  $EXPERIMENT"
echo "Job ID:      ${SLURM_JOB_ID:-local}  (array $ARR_ID)"

module load conda/latest
conda activate maniptrans

unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CPP
unset CMAKE_ARGS CMAKE_PREFIX_PATH CONDA_BUILD_SYSROOT
unset AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP READELF
unset GCC GXX GCC_AR GCC_NM GCC_RANLIB
unset CC_FOR_BUILD CXX_FOR_BUILD NVCC_PREPEND_FLAGS
unset DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXX=/usr/bin/g++
export SETUPTOOLS_USE_DISTUTILS=stdlib

# wandb is disabled for the partition sweep — too many concurrent tasks
# race on ./wandb's auto-resume id and crash before training starts.
# Set WANDB_MODE=disabled as a belt-and-braces in case any code path still
# touches the wandb client even with wandb_activate=False.
export WANDB_MODE=disabled

nvidia-smi -L

python main/rl/train.py \
    task=ResDexHand \
    dexhand=inspire \
    side=RH \
    headless=true \
    num_envs=4096 \
    learning_rate=2e-4 \
    test=false \
    randomStateInit=false \
    rh_base_model_checkpoint=assets/imitator_rh_inspire.pth \
    lh_base_model_checkpoint=assets/imitator_lh_inspire.pth \
    "dataIndices=[${INDICES}]" \
    actionsMovingAverage=0.4 \
    "experiment=${EXPERIMENT}" \
    wandb_activate=False
