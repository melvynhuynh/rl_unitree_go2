# NB03 stairs: progress log

Working notes for `notebooks/03_stairs.ipynb`, kept up to date at the end of each phase so work can resume from here.
Run log: `results/nb03_stairs_training_runs.csv` (created by `scripts/summarize_runs.py` at the first stairs run).

## Plan

| Phase | Content | Status |
| --- | --- | --- |
| A | Stairs MDP written, checked in the viewer with a random policy | done |
| B | Run 1 "can it climb?": flat MDP + terrain scan + relaxed rewards, `USE_NOISE = False`, `USE_DR = False` (variant `scan`) | training on gnoto |
| C | Run 2 final: actor noise + base CoM randomization (variant `scan_noise_dr`), compare with run 1 | to do |
| D | Only if needed: one single change based on the observed failure mode | to do |
| E | Demo config, CSV (`kept_for_submission`), roadmap boxes, oral points | to do |

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

| Run | Variant | Status | Level (CSV) | Ep. length | Lin tracking | Falls (base / orient.) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-12_17-53-05 | scan | complete | 5.92 | 418 | 0.38 | 0.15 / 0.14 | run 1 |

Reading run 1:
- Episodes end at the tile edge after 418 steps (8.4 s), consistent with 5.7 m at about 0.7 m/s. Episode_Reward terms are averaged over the 20 s maximum, so 0.38 lin tracking over 8.4 s is about 0.91 per second, close to flat (0.94).
- Mean level 5.9 is near the ceiling of about 6 that the wrap-around to a random row allows.
- Falls still end about 30 % of training episodes (stochastic policy, action std 0.96, robots kept at the limit of their ability by the curriculum). To check with the per-type levels: up or down?
- Viewer: climbs 20 cm steps and walks straight. Descent not checked yet.

## Viewer observations

Phase A, random policy, flags off (3.1i):
- spawn correct: top platform of the pyramids, bottom of the inverted pyramids, no robot inside a step;
- the terrain scan rays follow the steps;
- fallen robots are reset;
- robots are demoted to level 0 (flat rows) after a few episodes, which confirms that the demotion works;
- "Reset Environment" in the viewer keeps the curriculum levels: rerun the cell to see the initial spawn again.

## Next step

Phase B on gnoto: run 1 (variant `scan`, seed 42, flags off), then `python scripts/summarize_runs.py`. Report:
- from TensorBoard at the end: `Curriculum/terrain_levels/mean`, `stairs_up`, `stairs_down`, and whether the mean level is still rising at iteration 500;
- `Train/mean_reward`, `Train/mean_episode_length`, `Episode_Reward/track_linear_velocity`;
- termination shares: `time_out`, `terrain_edge_reached`, `base_contact`, `bad_orientation`;
- reward terms that dominate the penalties (`Episode_Reward/*`);
- viewer (3.4, up and down at 20 cm and at a lower height): does it climb and descend, and how it fails (trips on the edges, refuses to move, slips, falls when stepping down).
