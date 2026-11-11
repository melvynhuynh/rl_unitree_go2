# Roadmap: Legged Robots DRL project (solo)

Goal: train the Unitree Go2 with PPO (mjlab) on flat ground, then on stairs,
analyse the results, and submit the project before the oral exam (December 15 to 17).

Work order: **setup, NB01 flat, NB02 analysis, NB03 stairs, submission**.

---

## Organisation

### What runs where
| Notebook | Machine | Why |
| --- | --- | --- |
| `00_setup_check` | gnoto (GPU) | check simulator and viewer |
| `01_flat_terrain` | gnoto (GPU) | training |
| `02_flat_analysis` | local PC (CPU) | only reads the files in `logs/` |
| `03_stairs` | gnoto (GPU) | training |

### Synchronisation
- **Code**: this GitHub repo (`rl_unitree_go2`). On gnoto: clone the repo, then `git pull` / `git push` every session.
- **Data**: `logs/` is ignored by git. Download it regularly from gnoto (zip) and copy it locally for NB02.
- Commit after every step that works (e.g. "flat baseline v1", "metrics: cost of transport").

### Rules
1. **Never leave the GPU idle**: a training runs while I code or write something else.
2. **Check in the viewer before training** (random policy): a mistake costs seconds instead of a full run.
3. **Change one thing at a time** between two runs.
4. **Run log**: write every run in `doc/journal.md` (variant, change, observation). Used for the [ANSWER] cells and the oral.
5. **Do not over-tune**: as soon as the baseline walks cleanly, freeze it.

---

## Course methods to apply

Source: lecture 4 (G. Bellegarda, "Trajectory Optimization, Machine Learning, and Bio-Inspired Control"),
lecture 5 (M. Hutter, "Legged Robots @ ETH"), and the project slides (`2026_lr_project_01.pdf`).
Cite these lectures in the [ANSWER] cells and at the oral to link choices back to the course.

### MDP design (lecture 4: "RL considerations")
The lecture splits the design space into three groups. Each choice must be justified:
| Group | What we choose | Where |
| --- | --- | --- |
| MDP design decisions | observation space, action space, reward function | NB01 1.1, NB03 3.1 |
| Algorithm | PPO, hyperparameters, network, **random seeds / trials** | NB01 1.2 (provided, keep fixed) |
| Environment parameters | simulator dynamics, control gains, control / env time step, noise, latency | `scene.py`, NB03 3.1b and 3.1f |

- **Reward structure recommended in lecture 4** (CPG-RL): track base velocity commands (vx, vy, wz),
  penalise the other base velocities (vz, wx, wy), penalise energy. Our baseline follows it:
  `track_*`, `lin_vel_z`, `ang_vel_xy`, plus torque / action rate penalties.
- **Joint space vs task space actions** (lecture 4): joint space forces the policy to learn the
  kinematics; task space is more sample efficient. The project uses joint position targets (PD),
  be ready to discuss this trade-off.
- **Seeds matter** (Henderson et al., "Deep RL that Matters", lecture 4): always train several seeds
  and compare the difference between configurations with the spread between seeds.

### Sim-to-real techniques (lecture 4 "General RL techniques", lecture 5 "Key findings")
| Technique | In the project |
| --- | --- |
| Privileged information (asymmetric actor-critic, teacher-student) | NB01 critic observations (foot contacts, air time) |
| Domain randomization (mass, inertia, friction, perturbations) | NB03 3.1f (centre of mass, friction) |
| Observation noise | NB03 3.1b |
| Curriculum learning (easy task first, then harder) | NB03 3.1g terrain levels |
| Reward shaping | NB01 / NB03 rewards (`feet_air_time`, relaxed penalties on stairs) |
| Exteroception combined with proprioception | NB03 height scan |
| Actuator modeling / system identification | not in the project: mention as missing in 2.1c |
| Proprioceptive history encoding | not in the project: mention as a possible improvement |

### Metrics from the course (lecture 4, AllGaits)
Gaits are compared with **cost of transport (CoT)**, **mean base angular velocity** and
**mean joint acceleration**. Use the same metrics in NB02 (CoT is required; base angular velocity
fits the "base stability metric of your own design").

### Controlled experiment candidates (pick one, NB01 1.5)
1. **Ablation: remove `base_lin_vel` from the actor.** The real Go2 has no direct linear velocity
   sensor (needs a state estimator). Question: how much does tracking degrade without it?
   Links to deployability (2.1c) and privileged information (lecture 5).
2. **Substitution: torque penalty replaced by an energy (power) penalty.** Lecture 4 recommends an
   energy term. Question: does it lower the cost of transport, and at what tracking cost?
3. **Ablation: remove `feet_air_time`.** Lecture 4 shows that without a structural prior (CPG) the
   same reward leads to pathological gaits. Question: does the gait degrade into shuffling?
4. **Sensitivity: action scale (e.g. 0.25 vs 0.5).** Confounder: it also changes the size of the
   action rate penalty.

---

## 1. Setup
- [x] Get a GitHub token (private repo) and clone `rl_unitree_go2` on gnoto
- [x] gnoto terminal: `bash gnoto.sh`
- [x] Select the **LR Project Kernel**
- [x] Run `00_setup_check.ipynb` (GPU, simulator, viewer OK)
- [ ] Read `src/lr_project/mdp/` (observations, rewards, terminations, events) to know the available building blocks
- [ ] Read Rudin et al. 2022, *Learning to Walk in Minutes* (reward and observation tables)

## 2. NB01: Flat terrain

### Baseline (1.1 to 1.4)
- [ ] 1.1a `make_commands`: vx, vy, wz ranges, standing robots, forward-only robots, resampling time
- [ ] 1.1b `make_observations`: actor (measurable on the real robot) / critic (plus privileged information)
- [ ] 1.1c `make_actions`: offset from the default pose, scale
- [ ] 1.1d `make_rewards`: linear velocity and yaw rate tracking, penalties (action rate, torques, vertical velocity, orientation...), feet air time
- [ ] 1.1e `make_terminations`: time-out (not a failure) plus fall detection
- [ ] 1.1f `make_events`: random initial pose, joint angles at reset
- [ ] 1.1h Viewer with a random policy
- [ ] 1.3 Training and TensorBoard (`mean_reward`, `episode_length`, reward terms)
- [ ] 1.4a Viewer: is the gait clean? Adjust one reward, retrain (2 to 4 iterations max)
- [ ] Freeze the baseline, train it with 3 seeds
- [ ] 1.4b Collect rollouts (`command_grid`)

### Controlled experiment (1.5)
- [ ] Choose **a single** change from the candidates in "Course methods to apply"
- [ ] 1.5a Write the [ANSWER] **before** training (question, change, confounders, prediction)
- [ ] 1.5b `make_variant_env_cfg` and `VARIANT_NAME`
- [ ] 1.5c Viewer for the variant
- [ ] 1.5d Train the variant with the same 3 seeds
- [ ] 1.5f Rollouts with the **same** `eval_name` as the baseline
- [ ] Back up `logs/flat/`

## 3. NB02: Analysis (local, CPU)

### `src/lr_project/metrics.py`
- [ ] `yaw_rate_error_per_step`, `yaw_rate_error`
- [ ] `mechanical_power`, `cost_of_transport`
- [ ] `total_vertical_grf`, `peak_vertical_grf`, `weight_support`
- [ ] `duty_factor`, `stride_frequency`, `stride_length`, `foot_clearance`
- [ ] `action_rate`
- [ ] Own base stability metric (e.g. mean base angular velocity or roll/pitch variance, as in lecture 4)
- [ ] Test every function on a rollout (plausible orders of magnitude)

### Notebook
- [ ] 2.1 Load the baseline data
- [ ] 2.1a Training figures (reward, episode length, terms) and [ANSWER]
- [ ] 2.1b Evaluation figures (tracking overall and per command, CoT, stability, GRF, contact plot) and [ANSWER]
- [ ] 2.1c [ANSWER] deployability (observations, actions, hardware measurements, sim-to-real)
- [ ] 2.2a [ANSWER] question, H0, H1, metrics
- [ ] 2.2b [ANSWER] methods (change, what is held fixed, seeds, confounders)
- [ ] 2.2c Load the variant, at least 3 baseline vs variant figures and [ANSWER]
- [ ] 2.2e [ANSWER] discussion (expected / unexpected, how conclusive, mechanism, next test)

## 4. NB03: Stairs
- [ ] 3.1a `make_commands`: forward only, realistic speeds on stairs
- [ ] 3.1b `make_observations`: add `height_scan`, noise on each term (actor only)
- [ ] 3.1c `make_actions`: scale large enough to lift the legs
- [ ] 3.1d `make_rewards`: start from the flat baseline, relax penalties that get in the way (vertical velocity...)
- [ ] 3.1e `make_terminations`: fall plus `terrain_edge_reached` (time-out or failure?)
- [ ] 3.1f `make_events`: spawn on the platform, domain randomization (centre of mass, friction)
- [ ] Viewer: spawn, scan rays, reset after a fall
- [ ] 3.3 Training, watch `Curriculum/terrain_levels`
- [ ] 3.4 Viewer: up and down, over many episodes; note the failure modes
- [ ] Back up `logs/stairs/`

## 5. Submission
- [ ] Re-run the 3 notebooks, all outputs visible
- [ ] Check the length limits of the [ANSWER] cells; every number quoted comes from a cell
- [ ] Viewer videos: flat baseline, variant, stairs up and down (compressed)
- [ ] 5-slide presentation: baseline, controlled experiment, stairs, videos
- [ ] Zip `group_xx_project.zip` (under 500 MB):
  - `01_flat_terrain.ipynb`, `02_flat_analysis.ipynb`, `03_stairs.ipynb`
  - `.pptx` presentation (or PDF with video links)
  - best run `logs/flat/<date>_<variant>/` (without `model_0.pt`)
  - best run `logs/stairs/<date>_<variant>/` (without `model_0.pt`)
  - `src/lr_project/metrics.py`
- [ ] Upload to Moodle
- [ ] Book an oral slot as soon as it is announced

## 6. Oral exam preparation
- [ ] Re-read the "Things to think about" of every notebook section
- [ ] Be able to justify every MDP choice (why this observation, this weight, this termination)
- [ ] Be able to explain MDP, PPO (actor-critic, clipping, GAE), curriculum, domain randomization, sim-to-real
- [ ] Review practicals 1 to 7 (numerical simulation, single leg control, Cartesian PD, CPG with Hopf oscillators, virtual model control)
- [ ] Prepare the "Possible exam questions" from lecture 4:
  - How are model-based and learning-based methods similar? How are they different?
  - When would you use a model-based controller over a learning-based one, and vice versa?
  - Designing a controller with trajectory optimization: what cost function and constraints?
  - Training a controller with deep RL: how do you structure the MDP (observations, actions, rewards)?
  - What are the benefits of integrating CPGs with RL?
  - What makes RL challenging, and what do you consider when applying it to a new problem?
- [ ] Know the sim-to-real recipe from lecture 5: actuator modeling, domain randomization, curriculum,
  privileged teacher-student training, proprioceptive history, exteroception
