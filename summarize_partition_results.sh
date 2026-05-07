#!/bin/bash
# Print the headline table across every (object, n) cell that has
# results/<obj>_n<n>_aggregate.json present. Run this after invoking
# eval_partition_results.sh for each cell you care about.
#
# Output columns:
#   object  n  policies  trajs  mean_steps  mean_ratio  sr  fr

set -eo pipefail
cd ~/code/humanoid/ManipTrans

python3 - <<'EOF'
import glob, json, os, re

files = sorted(glob.glob("results/*_n*_aggregate.json"))
if not files:
    print("No results/<obj>_n<n>_aggregate.json found. Run eval_partition_results.sh first.")
    raise SystemExit(0)

rows = []
for f in files:
    with open(f) as fp:
        d = json.load(fp)
    rows.append(d)

# sort: cup, spoon, stick; then n = 1, 5, 10, all
obj_order = {"cup": 0, "spoon": 1, "stick": 2}
def n_key(n):
    return (1 if n == "all" else 0, int(n) if n != "all" else 0)
rows.sort(key=lambda r: (obj_order.get(r["object"], 99), n_key(r["n"])))

print(f"{'object':<7s} {'n':>4s} {'policies':>9s} {'trajs':>6s} "
      f"{'mean_steps':>11s} {'mean_ratio':>11s} {'sr':>6s} {'fr':>6s}")
for r in rows:
    print(f"{r['object']:<7s} {str(r['n']):>4s} {r['n_chunks']:>9d} "
          f"{r['n_trajectories']:>6d} {r['mean_steps']:>11.1f} "
          f"{r['mean_progress_ratio']:>11.3f} {r['success_rate']:>6.2f} "
          f"{r['failure_rate']:>6.2f}")
EOF
