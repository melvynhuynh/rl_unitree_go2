# NB03 stairs: progress log

Working notes for `notebooks/03_stairs.ipynb`, kept up to date at the end of each phase so work can resume from here.
Run log: `results/nb03_stairs_training_runs.csv` (created by `scripts/summarize_runs.py` at the first stairs run).

## Plan

| Phase | Content | Status |
| --- | --- | --- |
| A | Stairs MDP written, checked in the viewer with a random policy | done |
| B | Run 1 "can it climb?": flat MDP + terrain scan + relaxed rewards, `USE_NOISE = False`, `USE_DR = False` (variant `scan`) | done: climbs and descends 20 cm |
| C | Run 2 final: actor noise + base CoM randomization (variant `scan_noise_dr`), compare with run 1 | done: same performance, kept for submission |
| D | Only if needed: one single change based on the observed failure mode | skipped: run 2 crosses 20 cm up and down, rare falls |
| E | Demo config, CSV (`kept_for_submission`), roadmap boxes, oral points | done |

## Phase A: MDP decisions

Starting point: the validated flat MDP of NB01 (1.1a to 1.1f). Only what the stairs require is changed.

| Part | Choice | Reason |
| --- | --- | --- |
| Commands | forward only, `lin_vel_x = (0.5, 1.0)` m/s | tracked well on flat ground, moderate for 20 cm steps; at 80 % tracking a robot still covers 8 m in 20 s, more than the 5.7 m to the tile edge, so it can be promoted |
| Commands | `rel_standing_envs = 0` | a robot with a zero command is never promoted or demoted, and it never reaches the steps (spawn on the flat platform) |
| Commands | one command per episode (`resampling_time_range = 1e3`) | the curriculum compares the walked distance with the command at the end of the episode |
| Observations | flat actor terms + `height_scan` (54 rays, clip (-1, 1.5) m) | exteroception to see the steps ahead; the clip only bounds the 5 m value of a missed ray |
| Noise (flag) | lin vel 0.1, ang vel 0.2, gravity 0.05, joint pos 0.01, joint vel 1.5, scan 0.05 m; none on command and last action | rough sensor accuracy (Rudin et al. levels, scan lowered to a few cm as suggested in the notebook); command and last action are known exactly |
| Critic | same terms without noise + foot contact and air time | asymmetric actor-critic: the critic only runs in simulation, a clean state gives a better value estimate |
| Critic terms | built by a function, not shared with the actor | mjlab sets `noise = None` on the terms of a group without corruption; shared objects would silently remove the actor noise |
| Actions | scale 0.25, as on flat ground | validated on flat; Rudin et al. climb stairs with 0.25; first fix if feet catch the step edges |
| Rewards | `lin_vel_z` -2.0 to -0.5, `flat_orientation` -1.0 to -0.2, rest unchanged | climbing needs vertical velocity (about 0.3 m/s) and a pitched base (lecture 4 reward structure: penalize vz, but not so much that it blocks the task) |
| Terminations | time-out, `terrain_edge_reached` as time-out, base contact, bad orientation 1 rad | reaching the edge is a success, so its value is bootstrapped |
| Events | spawn +-0.3 m on the 1.5 m platform, any yaw, joints reset to default | all feet stay on the platform (0.3 + 0.35 < 0.75 m) |
| DR (flag) | base CoM offset, `startup`, `add`, x +-5 cm, y +-3 cm, z +-3 cm | one body per robot, like a fixed payload; larger x shifts overload thighs already near torque saturation on flat ground (22.7 of 23.5 N m) |

## Facts about the provided code (mjlab 1.6.0)

- Forward robots: `vx = |vx|.clamp(min=0.3)`, `vy = wz = 0`. Heading control is off by default.
- Terrain tile 12 m, 7 steps of 0.3 m around a 1.5 m platform, step height 0 to 20 cm over 10 rows. Pyramid spawn is on the top platform (walk down), inverted pyramid spawn is at the bottom of the pit (walk up).
- Curriculum: promotion if distance > 6 m or edge reached (the 80 % rule in the notebook text is not what the code does); demotion if distance < 0.5 x |command| x elapsed time.
- Initial levels are uniform over the 10 rows (mean about 4.5). A robot promoted past the last row goes back to a random row: a policy that succeeds everywhere settles around a mean level of 6, not 9. Compare runs with the mean level and the per-type levels (`stairs_up`, `stairs_down`), not against 9.
- The `terrain_level` column of the run log takes the first TensorBoard tag matching `Curriculum/terrain_levels.*`; check in TensorBoard which one it is (mean, max or a per-type level) and report the mean and per-type values separately.
- Mean episode length is lower than on flat ground even for a good policy: reaching the tile edge ends the episode early.
- `body_com_offset` with `"add"` adds to the default value, so it does not accumulate across calls.

## Notebook hygiene

- Flags and variant name in the "Run configuration" cell (`VARIANT = "scan" + "_noise" + "_dr"`), `SEEDS = [42]`.
- TensorBoard, training, loading and viewer are separate cells. The training cell skips a variant and seed that already has `model_499.pt`.
- Demo viewer: one robot, `episode_length_s = 1e6`, one staircase type (`DEMO_DIRECTION`) and a fixed step height on every row (`DEMO_STEP_HEIGHT`).

## Runs

| Run | Variant | Status | Mean level (TB, 499) | Ep. length | Lin tracking per s | Falls per step (base / orient.) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-12_17-53-05 | scan | complete | 5.89 | 418 | 0.92 | 0.08 / 0.08 | run 1 |
| 2026-11-12_20-22-16 | scan_noise_dr | complete | 5.99 | 429 | 0.89 | 0.04 / 0.12 | run 2 |

The `terrain_level` column of the CSV takes the first matching Curriculum tag, which is not the same for every run (5.92 for run 1, 9.0 = max for run 2): use the TensorBoard values above instead.

Reading run 1:
- Episodes end at the tile edge after 418 steps (8.4 s), consistent with 5.7 m at about 0.7 m/s. Episode_Reward terms are averaged over the 20 s maximum, so 0.38 lin tracking over 8.4 s is about 0.91 per second, close to flat (0.94).
- Episode_Termination values are counts per step, not fractions (the CSV columns `term_*` too). At iteration 499: edge 9.62, base contact 0.08, bad orientation 0.08, time-out 0.04, so about 2 % of episodes end in a fall.
- Curriculum (TensorBoard):

| Tag | it 0 | 125 | 250 | 375 | 499 |
| --- | --- | --- | --- | --- | --- |
| mean | 4.53 | 5.61 | 6.81 | 5.72 | 5.89 |
| stairs_up | 4.60 | 4.37 | 8.10 | 5.97 | 5.93 |
| stairs_down | 4.46 | 6.85 | 5.51 | 5.47 | 5.85 |
| max | 9 | 9 | 9 | 9 | 9 |

- Both types end at the ceiling of about 6 set by the wrap-around to a random row: the curriculum is saturated, up and down. The peaks above 6 (stairs_up 8.1 at 250) are the initial wave of robots climbing together before passing row 9. Not rising at 500 because it cannot go higher, so more iterations are not needed for the curriculum.
- Viewer: climbs and descends 20 cm steps smoothly and walks straight (see below).

Reading run 2 and comparison with run 1:

| | Run 1 (`scan`) | Run 2 (`scan_noise_dr`) |
| --- | --- | --- |
| mean level at 499 | 5.89 | 5.99 |
| stairs_up | 5.93 | 5.98 |
| stairs_down | 5.85 | 6.00 |
| edge reached per step | 9.62 | 8.54 |
| falls per step (base + orientation) | 0.16 | 0.16 |
| falls per episode | about 1.6 % | about 1.8 % |
| episode length | 418 (8.4 s) | 429 (8.6 s) |
| lin tracking per second of episode | 0.92 | 0.89 |
| yaw tracking per second (max 0.5) | 0.44 | 0.42 |

Run 2 curriculum (TensorBoard): mean 4.52, 5.40, 6.89, 5.66, 5.99 at iterations 0, 125, 250, 375, 499; same wave shape as run 1, saturated at the end.

- Noise and CoM randomization cost almost nothing in this setting: same saturated curriculum for both types, same fall rate, tracking 3 % lower and episodes 3 % longer (the robot walks slightly slower than commanded).
- What they bring cannot be measured here, since the evaluation uses the same simulator: robustness to sensor errors and to a shifted payload, i.e. a better chance to transfer to the real robot. Run 2 is the more deployable policy for the same performance, so it is the final one.

## Viewer observations

Phase A, random policy, flags off (3.1i):
- spawn correct: top platform of the pyramids, bottom of the inverted pyramids, no robot inside a step;
- the terrain scan rays follow the steps;
- fallen robots are reset;
- robots are demoted to level 0 (flat rows) after a few episodes, which confirms that the demotion works;
- "Reset Environment" in the viewer keeps the curriculum levels: rerun the cell to see the initial spawn again.

Run 1 (`scan`), demo at 20 cm on every row (3.4):
- up: climbs out of the pit and walks straight to the tile edge, then is reset in the next identical pit;
- down: walks down the pyramid smoothly, no fall over repeated crossings; the base pitches strongly nose down and the front legs reach for the next step;
- answer to "can it climb?": yes, up and down at the hardest height, without noise or randomization.

Run 2 (`scan_noise_dr`), demo at 20 cm on every row (3.4):
- up and down: crosses the stairs and walks straight, falls are rare (consistent with about 2 % of training episodes ending in a fall);
- same posture as run 1 when stepping down (base pitched nose down);
- demo videos up and down recorded from this run, for the presentation.

Training table of run 2 at iteration 499 (output of the training cell, so these numbers are in the notebook): mean reward 10.98, episode length 438, action std 0.95; Episode_Reward track_linear_velocity 0.39, feet_air_time 0.22, track_angular_velocity 0.18, action_rate -0.12, joint_acc -0.049, ang_vel_xy -0.041, lin_vel_z -0.027, flat_orientation -0.005, joint_torques -0.003, leg_contacts -0.003.

## Phase E: submission state

- Final stairs run: `logs/stairs/2026-11-12_20-22-16_scan_noise_dr/` (run 2), `kept_for_submission = yes` in the CSV. Remove `model_0.pt` before zipping.
- Backups outside the repo, in `rl_project/`: `flat_logs_2026-11-12.tar.gz`, `stairs_run1_scan.tar.gz`, `stairs_run2_scan_noise_dr.tar.gz`.
- The assignment (`doc/assignment.md`) says written text goes only in `[ANSWER]` cells and every number quoted must come from a cell. NB03 has no `[ANSWER]` cell, so the notebook keeps the template markdown unchanged and the design choices are explained in code comments only. The run 1 vs run 2 comparison and the oral points stay in this file and in the presentation.
- The notebook in the repo is the one executed on gnoto for run 2 (all outputs saved), with comments added afterwards. Re-running is not needed; if it is re-run, the training cell skips run 2 and the viewer cells must be stopped with the interrupt button.

## Oral points (Things to think about, NB03)

3.1a Commands
- Realistic speeds on stairs: 0.5 to 1.0 m/s. The flat policy tracked up to 1.0 m/s; at 0.5 m/s and 80 % tracking the robot still covers 8 m in 20 s, more than the 5.7 m to the tile edge, so the curriculum can promote it. Slower robots would time out without being promoted. The robot walks slightly slower than commanded (tracking 0.89 per second of episode).
- Standing still on stairs: not trained (`rel_standing_envs = 0`). A robot with a zero command is never promoted or demoted, and it spawns on the flat platform, so it would never stand on a step. On a real robot it would be useful (stopping on a staircase); it would need spawns on the steps or a command change in the middle of the stairs, and a curriculum that does not rely on the walked distance.

3.1b Observations
- Blind locomotion: possible with proprioception only, and with a history of observations the policy can feel a step after touching it (Lee et al. 2020, lecture 5). It reacts instead of anticipating: higher swing everywhere, more tripping on the first step, slower and more conservative gait. Here the actor is a memoryless MLP, so without the scan it would not know that a step is there.
- Noise on all observations: no. The command and the last action are known exactly (set by the operator, computed by the policy); noise on them makes learning harder without matching any real error. Noise goes on what sensors measure: IMU, encoders, state estimate, elevation map.
- Noisy critic: no. The critic only runs in simulation during training, so it can use the true state (asymmetric actor-critic, privileged information, lecture 5); a clean value estimate gives lower variance advantages. Implementation detail: actor and critic terms are built separately, because mjlab removes the noise of the terms of a group without corruption, which would also remove it from the actor if the term objects were shared.
- Terrain scan on a real robot: an elevation map built from depth cameras or a lidar and fused over time with the state estimate (robot-centric elevation mapping on ANYmal). Error sources: drift of the state estimate, occlusions behind steps, reflections; hence a noise of a few cm. Lecture 5: exteroception combined with proprioception, with a belief that falls back on proprioception when the map is wrong (Miki et al. 2022).

3.1c Actions
- The flat scale of 0.25 rad was enough: the robot climbs 20 cm steps. The Gaussian policy output is not bounded to [-1, 1], so larger joint offsets remain reachable. A larger scale would have been the first fix if the feet caught the step edges (leg_contacts stays at -0.003).

3.1d Rewards
- Reduced: `lin_vel_z` from -2.0 to -0.5 (climbing needs about 0.3 m/s of vertical velocity) and `flat_orientation` from -1.0 to -0.2 (the base pitches to follow the stairs). After training they are small (-0.027 and -0.005), so they no longer block the task.
- What dominates: the tracking terms (0.39 linear, 0.18 yaw) and feet air time (0.22). The largest penalty is the action rate (-0.12), then joint acceleration (-0.049) and roll and pitch rate (-0.041).
- The yaw rate command is always 0, so `track_angular_velocity` keeps the robot walking straight.

3.1e Terminations
- `terrain_edge_reached` is a time-out, not a failure: reaching the edge is a success, and as a time-out PPO bootstraps the value of the last state instead of treating it as a dead end. As a failure, the robot would learn to avoid the edge.
- Falls: base contact and tilt above 1 rad, as on flat ground. The base pitch on 20 cm x 30 cm stairs stays well below this limit.

3.1f Events and domain randomization
- When to sample: at startup, once per robot (4096 different bodies), like a real robot whose payload is fixed during a run. Sampling at reset gives more variety over training at some simulation cost; sampling during the episode would model a payload that moves, which is not the case here.
- How large: x +-5 cm, y and z +-3 cm, the size of a shifted battery or payload. Larger shifts in x load the front or rear legs, whose thighs were already near torque saturation on flat ground (22.7 of 23.5 N m); beyond that the task becomes impossible for some robots, which then only teach the policy to fall. Run 2 shows that these ranges cost almost nothing (same levels and falls, tracking 3 % lower).
- Other quantities worth randomizing: foot friction, base and payload mass, motor strength and PD gains, actuator latency (lecture 5: actuator network), external pushes, joint offsets, terrain height noise, step dimensions.
- Initial pose: spawn within +-0.3 m of the centre of the 1.5 m platform, any yaw, so all feet stay on the platform and the robot meets the steps from every direction.

3.1g Curriculum
- Walked distance instead of reward: the distance measures task success directly (crossing the stairs). The reward mixes tracking and penalties, depends on the weights, and a robot can collect reward without progressing (stepping in place, slow and smooth). The distance also does not depend on the reward tuning.
- Standing robots: a distance of 0 is never above the promotion threshold and never below the demotion threshold (half of a zero command times the time), so they stay at their level and add samples without curriculum signal. This is why the training uses no standing robots.
- Other curricula: on the command (start slow, then widen the speed range), on the noise and randomization ranges (start clean, then increase), on other terrains (slopes, rough ground, gaps), on the reward weights (raise the penalties once the robot walks), or a game-inspired automatic curriculum per terrain level as in Rudin et al.
- Reading the curve: a robot promoted beyond row 9 goes back to a random row, so a perfect policy settles around a mean level of 6, not 9. Both runs reach this ceiling for stairs up and down.

Run 1 vs run 2 (for the presentation)

| | Run 1 (`scan`) | Run 2 (`scan_noise_dr`) |
| --- | --- | --- |
| mean level at 499 | 5.89 | 5.99 |
| stairs up / down | 5.93 / 5.85 | 5.98 / 6.00 |
| episodes ending in a fall | about 1.6 % | about 1.8 % |
| episode length | 418 (8.4 s) | 429 (8.6 s) |
| lin tracking per second of episode | 0.92 | 0.89 |

Noise and DR cost little in simulation; their benefit is robustness on the real robot, which this simulator cannot measure. Run 1 values come from TensorBoard and the CSV, not from a notebook cell: quote them in the slides and at the oral, not in the notebook.

## Next step

NB03 is finished. Remaining items are in `doc/roadmap.md`, section 5 (submission): compressed videos, 5-slide presentation, zip with `logs/stairs/2026-11-12_20-22-16_scan_noise_dr/` without `model_0.pt`.
