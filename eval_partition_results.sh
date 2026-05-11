#!/bin/bash
# Aggregate per-trajectory eval stats across all partitioned policies for
# a given (object, n) sweep cell.
#
# Discovers every run matching:
#   runs/oakink2_inspire_<object>_n<n>_chunk*_*/
# Runs eval_oakink2_stage2.sh on any that don't yet have eval_progress_stats.json,
# then concatenates the per_traj rows from every chunk's JSON and prints a
# single overall summary.
#
# Usage:
#   ./eval_partition_results.sh <object> <n>
#
#   <object> : cup | spoon | stick
#   <n>      : 1, 5, 10, all (matches what was passed to submit_train_partition_array.sh)
#
# Output: prints aggregate to stdout AND dumps the merged JSON to:
#   results/<object>_n<n>_aggregate.json

set -eo pipefail

OBJECT="${1:?usage: $0 <cup|spoon|stick> <n|all>}"
N_ARG="${2:?usage: $0 <cup|spoon|stick> <n|all>}"

cd ~/code/humanoid/ManipTrans
mkdir -p results logs/eval_outputs

PATTERN="runs/oakink2_inspire_${OBJECT}_n${N_ARG}_chunk*_*/"
RUNS=( $(ls -d ${PATTERN} 2>/dev/null) )
if [ "${#RUNS[@]}" -eq 0 ]; then
    echo "No runs matching ${PATTERN}." >&2
    exit 1
fi
echo "Found ${#RUNS[@]} chunk run(s) for object=${OBJECT} n=${N_ARG}"

# Run eval on any that haven't been evaluated yet.
for run in "${RUNS[@]}"; do
    json="${run}eval_progress_stats.json"
    if [ -f "$json" ]; then
        echo "  [skip] $(basename $run) already has eval JSON"
        continue
    fi
    # check that a checkpoint exists (training might have been preempted before save)
    expt=$(grep -E "^experiment:" "${run}config.yaml" | head -1 | awk '{print $2}')
    ckpt="${run}nn/${expt}.pth"
    if [ ! -f "$ckpt" ]; then
        echo "  [skip] $(basename $run) has no checkpoint at ${ckpt}"
        continue
    fi
    echo "  [eval] $(basename $run)"
    ./eval_oakink2_stage2.sh "$run" > "logs/eval_outputs/$(basename $run).out" 2>&1 \
        || echo "    EVAL FAILED for $(basename $run) (see logs/eval_outputs/$(basename $run).out)"
done

# Aggregate every chunk's per_traj into one summary.
OUT="results/${OBJECT}_n${N_ARG}_aggregate.json"
python3 - <<EOF > >(tee /tmp/agg_${OBJECT}_n${N_ARG}.txt)
import glob, json, os, statistics

pat = "runs/oakink2_inspire_${OBJECT}_n${N_ARG}_chunk*_*/eval_progress_stats.json"
jsons = sorted(glob.glob(pat))

all_rows = []
chunks_seen = set()
for j in jsons:
    with open(j) as f:
        d = json.load(f)
    chunks_seen.add(os.path.basename(os.path.dirname(j)))
    all_rows.extend(d["per_traj"])

if not all_rows:
    print(f"WARN: no per_traj rows aggregated for ${OBJECT} n=${N_ARG}")
    raise SystemExit(0)

n_traj = len(all_rows)
n_chunks = len(chunks_seen)
mean_steps = statistics.mean(r["progress"] for r in all_rows)
mean_ratio = statistics.mean(r["ratio"] for r in all_rows)
sr = sum(r["success"] for r in all_rows) / n_traj
fr = sum(r["failure"] for r in all_rows) / n_traj
to = 1.0 - sr - fr

print()
print(f"=== Aggregate: object=${OBJECT} n=${N_ARG} ===")
print(f"  policies (chunks):   {n_chunks}")
print(f"  trajectories total:  {n_traj}")
print(f"  mean steps:          {mean_steps:.1f}")
print(f"  mean progress ratio: {mean_ratio:.3f}")
print(f"  success rate:        {sr:.3f}")
print(f"  failure rate:        {fr:.3f}")
print(f"  timeout rate:        {to:.3f}")

out = {
    "object": "${OBJECT}",
    "n": "${N_ARG}",
    "n_chunks": n_chunks,
    "n_trajectories": n_traj,
    "mean_steps": mean_steps,
    "mean_progress_ratio": mean_ratio,
    "success_rate": sr,
    "failure_rate": fr,
    "timeout_rate": to,
    "per_traj": all_rows,
}
with open("${OUT}", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[agg] wrote ${OUT}")
EOF
