The final experiment deliverable is a table of [column - benchmark, row - tuning set size, data - difference between tuning set and evaluation set] for bm25 and kb, tuning set size [25, 50, 75, 100], where the difference is defined to be the relative difference of speedup. For instance, if under QoS loss <= 3%, tuning set can have 1.2x speedup and evaluation set can have 1.4x, the difference of speedup is (1.4x - 1.2x) / 1.2x. Note that the / 1.2x is important.

To complete this experiment we already made tuning set generation automated (our v6 version). We prototyped the tuning set size = 100 in v5 version, and we decide not to repeat it for budget purpose so the v6's tuning set size = 100 is just v5's data. 

Please first copy the static and dynamic-v6 to plot directory, and we will work on the plot directory afterwards.

From our previous runs, we can see that 500 as evaluation size is too large which makes evaluation running very slowly. This motivates us to change it to 250 and getting rid of the warmup run for evaluation. Both are done. 

Evaluation requires a replay set on evaluation mode. Making the replay set requires 2 things: 
- the tuning set for both dynamic and static 
- the set of QoS loss threshold after deduplication detailed below

First, you will generate merged csvs, plot them on all v6 data to examine the tuning set. Then you will follow the following instruction to generate the replay configuration csvs for each tuning set size.


Replay csv generation instruction (for each tuning set size):
```
Now I'm going to use the configurations in v6 to make a replay set for evaluation for both benchmarks.

Here is how you are going to pick:
1. first find the exact configuration in the static csv, obtaining the exact acuracy.
2. QoS loss is defined to be exact accuracy - the accuracy of the configuration you are picking. 
3. Pick QoS loss = 0.01, 0.02, ..., 0.09, find the best configuration (in terms of performance) that has QoS loss <= the target QoS loss.
4. If duplication exists, pick the second best, third best, etc. until you find a unique configuration for each QoS loss target. Duplication means the case when the same configuration is picked for multiple QoS loss targets.

Your output:

* replay csvs for evaluation (for each benchmark, one from static csv and one from dynamic csv)
* command lines to run the evaluation using the submit script and slurm script in slurm bench.
```

## Generate KB Dynamic/Static Difference Tables

This section defines how to generate the two KB tables:
- table 1: dynamic
- table 2: static

Both tables use:
- rows: tuning set size `[25, 50, 75, 100]`
- columns: QoS loss constraints `<= 3%`, `<= 6%`, `<= 9%`
- cell value: relative difference of speedup (in percent)

Formula:
- `speedup = exact_time / best_time_under_qos`
- `relative_difference = (speedup_eval - speedup_tune) / speedup_tune`
- report as percentage: `100 * relative_difference`

### Data Sources (Generic)

Do not hardcode job IDs. Resolve files by convention + run manifest.

1. Tuning merged CSVs (from plot pipeline):
   - dynamic: `plot/<benchmark>_dynamic_v<version>_s<size>.csv`
   - static: `plot/<benchmark>_static_v<version>_s<size>.csv`
   - example benchmark values: `kb`, `bm25`
   - example sizes: `25`, `50`, `75`, `100`

2. Evaluation replay result CSVs (from SLURM replay runs):
   - filename pattern: `logs/llm_<benchmark>_replay_eval_*.csv`
   - map each file to `(benchmark, mode, size)` using one of:
     - a manifest generated at submit time, or
     - replay config filename embedded in log, or
     - deterministic submission order recorded by the run script.

3. Recommended manifest format (one row per submitted replay job):
   - columns: `benchmark,mode,size,config_csv,jobid,result_csv`
   - this makes post-processing fully reproducible even after recollection.

Important:
- Exact baseline for each split is taken from the static exact configuration (all knobs zero) in that split's static CSV.
- Dynamic table uses dynamic candidate configs only.
- Static table uses static candidate configs only.

### Concrete Strategy

For each benchmark `b` and each tuning size `s in {25,50,75,100}`:

1. Load tuning merged CSVs:
   - `tune_dynamic = plot/{b}_dynamic_v{version}_s{s}.csv`
   - `tune_static = plot/{b}_static_v{version}_s{s}.csv`

2. Compute tuning exact baseline from `tune_static`:
   - find row(s) where all knob columns are `0`
   - `tune_exact_time = mean(time_ms of exact rows)`
   - `tune_exact_acc = mean(accuracy of exact rows)`

3. Resolve evaluation replay CSVs for this benchmark+size:
   - `eval_dynamic` from manifest row where `mode=dynamic`
   - `eval_static` from manifest row where `mode=static`

4. Compute evaluation exact baseline from `eval_static`:
   - find row(s) where all knob columns are `0`
   - `eval_exact_time = mean(time_ms of exact rows)`
   - `eval_exact_acc = mean(accuracy of exact rows)`

5. For each QoS threshold `q in {0.03,0.06,0.09}`:
   - tuning candidates satisfy: `tune_exact_acc - acc <= q`
   - evaluation candidates satisfy: `eval_exact_acc - acc <= q`
   - dynamic table:
     - use dynamic candidates from `tune_dynamic` and `eval_dynamic`
   - static table:
     - use static candidates from `tune_static` and `eval_static`
   - for each table cell:
     - `best_time_tune = min(time_ms among tuning candidates)`
     - `best_time_eval = min(time_ms among evaluation candidates)`
     - `speedup_tune = tune_exact_time / best_time_tune`
     - `speedup_eval = eval_exact_time / best_time_eval`
     - `diff_percent = 100 * (speedup_eval - speedup_tune) / speedup_tune`

6. Fill two tables:
   - dynamic table: use dynamic tuning/eval candidates
   - static table: use static tuning/eval candidates

7. If no candidate satisfies a QoS threshold for a table cell, mark `NA`.
