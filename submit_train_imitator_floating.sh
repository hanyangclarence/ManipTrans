#!/bin/bash
#SBATCH --job-name=maniptrans-imit-float
#SBATCH --output=logs/slurm_outputs/slurm-%j.out
#SBATCH --error=logs/slurm_outputs/slurm-%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a16

# Stage 1 imitator on a floating dexhand (no fixed arm). Selects the original
# tuned Kp/Kd + reward weights baked into each dexhand class. Works on the
# Manus fingertip-only pipeline (DexHandImitatorManus, useFingertipsOnly=True);
# weight_idx in inspire/shadow/xhand/wujihand all expose the same 5 *_tip keys,
# so no pickle re-conversion is needed when swapping hands. Default is
# `inspire` — the ManipTrans canonical hand whose reward weights and PD gains
# the upstream paper actually tuned against.
#
# Usage:
#   sbatch submit_train_imitator_floating.sh                                   # default: inspire on all marker_pen
#   sbatch submit_train_imitator_floating.sh shadow   marker_pen 1             # single trajectory
#   sbatch submit_train_imitator_floating.sh xhand    hammer_mixed 10          # first 10
#   sbatch submit_train_imitator_floating.sh wujihand                          # wuji-only (matches retargeting pickle)

set -eo pipefail

DEXHAND="${1:-inspire}"
DATASET="${2:-marker_pen}"
N_TRAJ_ARG="${3:-all}"

case "$DEXHAND" in
    inspire|wujihand|shadow|xhand) ;;
    *)
        echo "Unsupported dexhand '$DEXHAND' — expected 'inspire', 'wujihand', 'shadow', or 'xhand'" >&2
        exit 1
        ;;
esac

case "$DATASET" in
    marker_pen)   ROOT="manus_marker_pen" ;;
    hammer_mixed) ROOT="manus_hammer_mixed" ;;
    *)
        echo "Unknown dataset '$DATASET' — expected 'marker_pen' or 'hammer_mixed'" >&2
        exit 1
        ;;
esac

if [ "$N_TRAJ_ARG" != "all" ] && ! [[ "$N_TRAJ_ARG" =~ ^[1-9][0-9]*$ ]]; then
    echo "n_traj must be 'all' or a positive integer, got '$N_TRAJ_ARG'" >&2
    exit 1
fi

cd ~/code/humanoid/ManipTrans

mkdir -p logs/slurm_outputs

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

export WANDB_API_KEY=28b3c634497c0dc6c16767729d4719b1012a94f2
export SETUPTOOLS_USE_DISTUTILS=stdlib

export WANDB_DIR="wandb_${SLURM_JOB_ID:-local}"
mkdir -p "$WANDB_DIR"

nvidia-smi

SEQ_DIR="data/${ROOT}/sequences"
if [ ! -d "$SEQ_DIR" ]; then
    echo "Sequence dir '$SEQ_DIR' not found — did you run convert_manus_to_maniptrans.py?" >&2
    exit 1
fi

if [ "$N_TRAJ_ARG" = "all" ]; then
    SELECTED=$(ls "$SEQ_DIR"/*.pkl)
    N_TAG="all"
else
    SELECTED=$(ls "$SEQ_DIR"/*.pkl | head -n "$N_TRAJ_ARG")
    N_TAG="n${N_TRAJ_ARG}"
fi

INDICES=$(echo "$SELECTED" \
    | xargs -n1 basename -s .pkl \
    | sed "s|^|manus@${ROOT}@|" \
    | paste -sd ,)

N_TRAJ=$(echo "$INDICES" | tr ',' '\n' | wc -l)
echo "Hand: ${DEXHAND}    Dataset: ${ROOT}    n_trajectories=${N_TRAJ} (requested=${N_TRAJ_ARG})"

SELECTED_PATHS="$SELECTED" python - <<'PY'
import os, pickle, sys
paths = [p for p in os.environ["SELECTED_PATHS"].splitlines() if p]
lens = [pickle.load(open(p, "rb"))["obj_trajectory"].shape[0] for p in paths]
print(f"  frames: min={min(lens)} max={max(lens)} mean={sum(lens)//len(lens)}")
if max(lens) > 2000:
    print(f"  WARNING: max length {max(lens)} > episodeLength=2000; bump ++task.env.episodeLength", file=sys.stderr)
PY

EXPERIMENT="imit_${DEXHAND}_${DATASET}_${N_TAG}_$(date +%Y%m%d_%H%M%S)"
echo "Run: ${EXPERIMENT}"

python main/rl/train.py \
    task=DexHandImitatorManus \
    dexhand=${DEXHAND} \
    side=RH \
    "dataIndices=[${INDICES}]" \
    num_envs=4096 \
    experiment="${EXPERIMENT}" \
    wandb_activate=True \
    wandb_project=maniptrans-manus \
    wandb_name="${EXPERIMENT}" \
    "wandb_tags=[imitator,${DEXHAND},${DATASET}]"
