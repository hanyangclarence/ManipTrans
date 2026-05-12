# OakInk2 Stage-2 Partition Sweep (WujiHand)

Train Stage-2 wuji floating-hand policies on partitioned trajectory chunks of
varying width `n`, then evaluate each policy on its own training chunk.
Compares "separate" (small n) vs "joint" (large n) training under a fixed
budget. (The inspire-baseline equivalent lives on branch
`oakink2-baseline-original`.)

## Pre-requisites

- `indices_<object>.txt` — one trajectory id per line (`<5hex>@<stage>`),
  produced by `filter_oakink2_indices.py`. Already present for cup (93),
  spoon (98), stick (13).
- All trajectories retargeted under
  `data/retargeting/OakInk-v2/mano2wujihand_rh/` (the user already produced
  1,930 of these; all 204 cup/spoon/stick indices are covered).
  `submit_retarget_oakink2_array.sh indices_<obj>.txt wujihand` re-runs any
  that go missing.
- **Stage-1 imitator checkpoint at `assets/imitator_rh_wujihand.pth`** —
  produced by `submit_train_stage1_oakink2_wuji.sh`. After that job finishes,
  copy the result into `assets/`:
  ```
  cp runs/imitator_oakink2_wuji_rh_all_*/nn/imitator_oakink2_wuji_rh_all_*.pth \
     assets/imitator_rh_wujihand.pth
  ```
- `runs/` is a symlink to `/scratch4/.../maniptrans_runs/` (set up already).

## How partitioning works

For an `(object, n)` cell with `M` trajectories in `indices_<object>.txt`:

```
total_chunks = ceil(M / n)
chunks       = min(total_chunks, 20)     # hard cap of 20 policies / cell
```

Array task `k` trains one policy on lines `[k*n+1 : (k+1)*n]` of the index
file. Any tail beyond `chunks * n` is dropped (the "subsample and skip"
rule). The last chunk may be shorter than `n` (e.g. spoon n=5 chunk 19 has
3 trajectories, since 98 = 19·5 + 3). `n=all` collapses to a single chunk
covering the entire file.

Each policy gets a unique experiment name embedding the chunk index + array
job id, so eval can glob all chunks of a sweep cell:

```
oakink2_wuji_<object>_n<n>_chunk<kk>_arr<SLURM_ARRAY_JOB_ID>
```

## 1. Launch training

Each cell is one `sbatch --array=...` submission. SLURM time cap is 6h per
policy (set in `submit_train_partition_array.sh`). The `%8` in the array
spec caps concurrent running tasks — adjust per cluster fairness.

```bash
# cup (M=93)
sbatch --array=0-19%8 submit_train_partition_array.sh cup 1     # 20 policies, 20/93 covered
sbatch --array=0-18%8 submit_train_partition_array.sh cup 5     # 19 policies, 93/93
sbatch --array=0-9%8  submit_train_partition_array.sh cup 10    # 10 policies, 93/93
sbatch --array=0-0    submit_train_partition_array.sh cup all   #  1 policy,   93/93

# spoon (M=98)
sbatch --array=0-19%8 submit_train_partition_array.sh spoon 1   # 20 policies, 20/98
sbatch --array=0-19%8 submit_train_partition_array.sh spoon 5   # 20 policies, 98/98
sbatch --array=0-9%8  submit_train_partition_array.sh spoon 10  # 10 policies, 98/98
sbatch --array=0-0    submit_train_partition_array.sh spoon all #  1 policy,   98/98

# stick (M=13)
sbatch --array=0-12%8 submit_train_partition_array.sh stick 1   # 13 policies, 13/13
sbatch --array=0-2    submit_train_partition_array.sh stick 5   #  3 policies, 13/13
sbatch --array=0-1    submit_train_partition_array.sh stick 10  #  2 policies, 13/13
sbatch --array=0-0    submit_train_partition_array.sh stick all #  1 policy,   13/13
```

Total: 12 array submissions, ~120 policies, 6h each.

Logs land in `logs/slurm_outputs/stage2-part-<arrjobid>_<task>.out`. wandb
runs go to `./wandb/run-<ts>-uid_<experiment>/`.

## 2. Evaluate one cell

After all chunks of a cell finish (or you `scancel` them):

```bash
./eval_partition_results.sh <object> <n>
# e.g.
./eval_partition_results.sh cup 5
./eval_partition_results.sh stick all
```

For each chunk run:
- If the run already has `eval_progress_stats.json`, skipped.
- Else, runs `eval_oakink2_stage2.sh <run_dir>` (deterministic policy, frame 0,
  one rollout per training trajectory).

Then aggregates every chunk's `per_traj` rows into one cell-level summary,
prints the headline numbers, and writes:

```
results/<object>_n<n>_aggregate.json
```

with per-trajectory rows (`data_idx`, `seq_len`, `progress`, `ratio`,
`success`, `failure`) plus the means/rates over all evaluated trajectories.

## 3. Summarize the whole sweep

After all 12 cells have been evaluated:

```bash
./summarize_partition_results.sh
```

Prints one row per `(object, n)` cell:

```
object    n  policies  trajs  mean_steps  mean_ratio    sr    fr
cup       1        20     20         …           …      …     …
cup       5        19     93         …           …      …     …
cup      10        10     93         …           …      …     …
cup     all         1     93         …           …      …     …
spoon     1        20     20         …           …      …     …
…
```

Plot `mean_progress_ratio` (or `sr`) vs `n` for each object to read off the
separate-vs-joint tradeoff.

## Caveats

- **n=1 covers only 20/93 of cup and 20/98 of spoon** because of the
  20-policy cap. The aggregate at n=1 is over a different trajectory subset
  than n=all — they're not strictly apples-to-apples. Eyeballing the
  curve still works; rigorous comparison would require evaluating all
  policies on the same held-out trajectory set.
- **6h per policy is a guess.** If the curve at n=all looks flat-bad, that
  may be under-training (single policy on 93 round-robin'd trajectories =
  ~44 envs per trajectory) rather than a fundamental ceiling. Re-launch
  the relevant cells with a higher SLURM `--time=` to check.
- **`runs/` is a symlink to scratch.** Don't `rm -rf runs` — that follows
  the symlink and wipes scratch storage. Use `find runs/ -mindepth 1 …`
  if you need to clean up specific runs.
- **`pack_data().squeeze()` quirk** means `num_envs == 1` crashes. The
  partition launcher avoids this since it always picks `num_envs=4096`
  for training. The eval launcher (`eval_oakink2_stage2.sh`) pads to
  `num_envs=2` when a chunk has only 1 trajectory.

## Files

- `submit_train_partition_array.sh` — SLURM array launcher (one task = one chunk).
- `eval_partition_results.sh` — runs eval on all chunks of one cell, aggregates.
- `eval_oakink2_stage2.sh` — single-run eval (called by the above).
- `summarize_partition_results.sh` — final cross-cell table.
- `indices_<object>.txt` — frozen trajectory list per object.
- `results/<object>_n<n>_aggregate.json` — per-cell aggregated stats.
