# Training run log

One CSV per experiment, **one row per training run** (one folder in `logs/<experiment>/`).
`logs/` is not versioned, so these files are the permanent record of every run.

| File | Produced by | Experiment in `logs/` |
| --- | --- | --- |
| `nb01_flat_terrain_training_runs.csv` | `notebooks/01_flat_terrain.ipynb` | `flat` |
| `nb03_stairs_training_runs.csv` | `notebooks/03_stairs.ipynb` | `stairs` (created at the first stairs run) |

CSV is used instead of xlsx so git can show the changes line by line; it opens directly in Excel.

## Updating

After each training run, on gnoto (repository root, venv active):

```bash
source ~/gnoto_venvs/lr-project/bin/activate
python scripts/summarize_runs.py
```

Then fill in the manual columns and commit the CSV.

## Columns

**Automatic** (recomputed from the TensorBoard logs at every call; "final" values are the mean of the last 20 iterations):

| Column | Meaning |
| --- | --- |
| `run_id` | run folder name `<date>_<time>_<variant>` |
| `experiment`, `variant`, `seed`, `created` | from the folder name and `params/agent.yaml` |
| `status` | `complete` (reached `max_iterations`), `interrupted`, or `logs_deleted` (folder removed, row kept as history) |
| `last_iteration`, `checkpoint` | last logged iteration and last saved policy |
| `mean_reward`, `mean_episode_length` | `Train/*` curves (1000 = no fall within 20 s) |
| `rew_track_lin_vel`, `rew_track_ang_vel` | tracking reward terms (max = their weight: 1.0 and 0.5) |
| `err_vel_xy`, `err_vel_yaw` | command tracking errors (m/s, rad/s) |
| `rew_action_rate`, `rew_feet_air_time`, `air_time_mean` | smoothness and stepping |
| `term_base_contact`, `term_bad_orientation` | failure terminations |
| `action_std` | policy exploration noise |
| `terrain_level` | stairs curriculum level (stairs only) |
| `eval_files` | evaluation rollouts saved in `eval/` |

**Manual** (never overwritten by the script):

| Column | What to write |
| --- | --- |
| `change_vs_baseline` | the single change compared to the baseline (`none` for the baseline) |
| `purpose` | why this run was trained (tuning, seed repeat, controlled experiment, ...) |
| `kept_for_submission` | `yes` for the 2 log folders submitted (best flat, best stairs) |
| `notes` | what you saw in the viewer, failures, decisions |
