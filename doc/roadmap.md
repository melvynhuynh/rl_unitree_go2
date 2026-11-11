# Roadmap — projet DRL Legged Robots (solo)

Objectif : entraîner le Unitree Go2 avec PPO (mjlab) sur sol plat puis sur escaliers,
analyser les résultats, et rendre le projet avant l'oral (15–17 décembre).

Ordre de travail : **setup → NB01 flat → NB02 analyse → NB03 stairs → rendu**.

---

## Organisation

### Où tourne quoi
| Notebook | Machine | Pourquoi |
| --- | --- | --- |
| `00_setup_check` | gnoto (GPU) | vérifier simulateur + viewer |
| `01_flat_terrain` | gnoto (GPU) | entraînement |
| `02_flat_analysis` | PC local (CPU) | lit seulement les fichiers de `logs/` |
| `03_stairs` | gnoto (GPU) | entraînement |

### Synchronisation
- **Code** : ce repo GitHub (`rl_unitree_go2`). Sur gnoto : cloner le repo, puis `git pull` / `git push` à chaque session.
- **Données** : `logs/` est ignoré par git. Le télécharger régulièrement depuis gnoto (zip) et le copier en local pour le NB02.
- Commit après chaque étape qui marche (ex. « baseline flat v1 », « metrics: cost of transport »).

### Règles
1. **Le GPU ne doit jamais attendre** : un entraînement tourne pendant que je code ou rédige autre chose.
2. **Vérifier dans le viewer avant d'entraîner** (politique random) : une erreur coûte quelques secondes au lieu d'un run.
3. **Changer une seule chose à la fois** entre deux runs.
4. **Journal des essais** : chaque run noté dans `doc/journal.md` (variant, changement, observation). Sert pour les [ANSWER] et l'oral.
5. **Ne pas sur-tuner** : dès que la baseline marche proprement, la figer.

---

## 1. Setup
- [ ] Récupérer le token GitHub (repo privé) et cloner `rl_unitree_go2` sur gnoto
- [ ] Terminal gnoto : `bash gnoto.sh`
- [ ] Sélectionner le kernel **LR Project Kernel**
- [ ] Exécuter `00_setup_check.ipynb` (GPU, simulateur, viewer OK)
- [ ] Lire `src/lr_project/mdp/` (observations, rewards, terminations, events) pour connaître les briques disponibles
- [ ] Lire Rudin et al. 2022, *Learning to Walk in Minutes* (tableaux rewards / observations)

## 2. NB01 — Flat terrain

### Baseline (1.1 → 1.4)
- [ ] 1.1a `make_commands` : plages vx, vy, ωz, robots à l'arrêt, robots en marche avant, rééchantillonnage
- [ ] 1.1b `make_observations` : actor (mesurable sur le vrai robot) / critic (+ infos privilégiées)
- [ ] 1.1c `make_actions` : offset par rapport à la pose par défaut, échelle
- [ ] 1.1d `make_rewards` : suivi vitesse linéaire + yaw, pénalités (action rate, couples, vitesses verticales, orientation…), feet air time
- [ ] 1.1e `make_terminations` : time-out (pas un échec) + détection de chute
- [ ] 1.1f `make_events` : pose initiale aléatoire, angles articulaires au reset
- [ ] 1.1h Viewer avec politique random
- [ ] 1.3 Entraînement + TensorBoard (`mean_reward`, `episode_length`, termes de reward)
- [ ] 1.4a Viewer : la démarche est-elle propre ? → ajuster une reward, ré-entraîner (2–4 itérations max)
- [ ] Figer la baseline, l'entraîner avec 3 seeds
- [ ] 1.4b Collecter les rollouts (`command_grid`)

### Expérience contrôlée (1.5)
- [ ] Choisir **un seul** changement (ablation, ajout, substitution ou sensibilité)
- [ ] 1.5a Rédiger l'[ANSWER] **avant** d'entraîner (question, changement, confounders, prédiction)
- [ ] 1.5b `make_variant_env_cfg` + `VARIANT_NAME`
- [ ] 1.5c Viewer du variant
- [ ] 1.5d Entraîner le variant avec les mêmes 3 seeds
- [ ] 1.5f Rollouts avec le **même** `eval_name` que la baseline
- [ ] Sauvegarder `logs/flat/`

## 3. NB02 — Analyse (local, CPU)

### `src/lr_project/metrics.py`
- [ ] `yaw_rate_error_per_step`, `yaw_rate_error`
- [ ] `mechanical_power`, `cost_of_transport`
- [ ] `total_vertical_grf`, `peak_vertical_grf`, `weight_support`
- [ ] `duty_factor`, `stride_frequency`, `stride_length`, `foot_clearance`
- [ ] `action_rate`
- [ ] Métrique personnelle de stabilité de la base (ex. variance roll/pitch)
- [ ] Tester chaque fonction sur un rollout (ordres de grandeur plausibles)

### Notebook
- [ ] 2.1 Charger les données de la baseline
- [ ] 2.1a Figures training (reward, episode length, termes) + [ANSWER]
- [ ] 2.1b Figures évaluation (tracking global et par commande, CoT, stabilité, GRF, contact plot) + [ANSWER]
- [ ] 2.1c [ANSWER] déployabilité (observations, actions, mesure sur hardware, sim-to-real)
- [ ] 2.2a [ANSWER] question, H0, H1, métriques
- [ ] 2.2b [ANSWER] méthodes (changement, éléments fixés, seeds, confounders)
- [ ] 2.2c Charger le variant, ≥ 3 figures baseline vs variant + [ANSWER]
- [ ] 2.2e [ANSWER] discussion (attendu / inattendu, conclusivité, mécanisme, suite)

## 4. NB03 — Stairs
- [ ] 3.1a `make_commands` : marche avant, vitesses réalistes sur escaliers
- [ ] 3.1b `make_observations` : + `height_scan`, bruit sur chaque terme (actor seulement)
- [ ] 3.1c `make_actions` : échelle suffisante pour lever les pattes
- [ ] 3.1d `make_rewards` : partir de la baseline flat, relâcher les pénalités gênantes (vitesse verticale…)
- [ ] 3.1e `make_terminations` : chute + `terrain_edge_reached` (time-out ou échec ?)
- [ ] 3.1f `make_events` : spawn sur la plateforme, domain randomization (centre de masse, frottement)
- [ ] Viewer : spawn, rayons du scan, reset après chute
- [ ] 3.3 Entraînement — suivre `Curriculum/terrain_levels`
- [ ] 3.4 Viewer : montée et descente, sur plusieurs épisodes ; noter les modes d'échec
- [ ] Sauvegarder `logs/stairs/`

## 5. Rendu
- [ ] Ré-exécuter les 3 notebooks, toutes les sorties visibles
- [ ] Vérifier les limites de longueur des [ANSWER] ; chaque chiffre cité vient d'une cellule
- [ ] Vidéos du viewer : flat baseline, variant, escaliers montée + descente (compressées)
- [ ] Présentation 5 slides : baseline, expérience contrôlée, stairs, vidéos
- [ ] Zip `group_xx_project.zip` (< 500 MB) :
  - `01_flat_terrain.ipynb`, `02_flat_analysis.ipynb`, `03_stairs.ipynb`
  - présentation `.pptx` (ou PDF + liens vidéos)
  - meilleur run `logs/flat/<date>_<variant>/` (sans `model_0.pt`)
  - meilleur run `logs/stairs/<date>_<variant>/` (sans `model_0.pt`)
  - `src/lr_project/metrics.py`
- [ ] Déposer sur Moodle
- [ ] Réserver un créneau d'oral dès l'annonce

## 6. Préparation de l'oral
- [ ] Relire les « Things to think about » de chaque section des notebooks
- [ ] Savoir justifier chaque choix du MDP (pourquoi cette observation, ce poids, cette termination)
- [ ] Savoir expliquer MDP, PPO (actor-critic, clipping, GAE), curriculum, domain randomization, sim-to-real
- [ ] Réviser les TPs 1–7 (simulation numérique, contrôle d'une patte, contrôle quadrupède)
