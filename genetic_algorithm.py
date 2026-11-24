from __future__ import annotations
import math
import multiprocessing
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import random
import statistics
import numpy as np
import torch
import pygame
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    GA_POPULATION_SIZE,
    GA_SIMULATION_STEPS,
    ES_SIGMA_START,
    ES_SIGMA_END,
    ES_LR_START,
    ES_LR_END,
    ES_WEIGHT_DECAY,
    MODEL_SAVE_PATH,
    SHEEP_COUNT,
    SHEEP_RADIUS,
    DOG_RADIUS,
    DOG_START_POS,
    HERD_MAX_DISTANCE,
    START_ZONE_RECT,
)
from map_env import PastureMap, SELECTABLE_ENV_IDS, create_map, resolve_env_id
from transformer_brain import HerdingTransformerBrain
from dog import HerdingDog
from sheep import Sheep


def _init_worker():
    #Prevent PyTorch OpenMP thread contention across parallel worker processes.
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    torch.set_num_threads(1)


def _place_clear_of_rocks(
    x: float,
    y: float,
    radius: float,
    map_env: PastureMap,
    bounds: tuple[float, float, float, float] | None = None,
) -> pygame.Vector2:
    #Nudges a spawn point outside any overlapping rock obstacle and keeps it within valid bounds.
    pos = pygame.Vector2(x, y)
    for _ in range(2):
        for rock in map_env.rocks:
            delta = pos - rock.pos
            min_d = rock.radius + radius + 5.0
            if delta.length_squared() < min_d * min_d:
                n = delta.normalize() if delta.length_squared() > 0.001 else pygame.Vector2(1.0, 0.0)
                pos = rock.pos + n * min_d
        if bounds is not None:
            bx1, bx2, by1, by2 = bounds
            pos.x = max(bx1, min(bx2, pos.x))
            pos.y = max(by1, min(by2, pos.y))
    return pos


def spawn_scenario(
    count: int,
    map_env: PastureMap,
    seed: int | None = None,
    scenario_mode: str = "full",
) -> tuple[list[Sheep], tuple[float, float, float]]:
    """
    Spawns a herd and returns `(herd, (dog_x, dog_y, dog_facing_angle))`
    adapted dynamically to `map_env`'s river, bridge, and goal-pen geometry:
      - "east":        Goal-bank pasture between bridge and gate (Stage 1 herding & penning drill)
      - "bridge":      Start bank approaching bridge (Stage 1 bridge funnel drill)
      - "full_center": Start pasture center
      - "full_north":  North start pasture
      - "full_south":  South start pasture
      - "full":        Randomized start pasture spawn for evaluation / play
    """
    rng = random.Random(seed)
    zone_x, _, _, _ = START_ZONE_RECT

    is_rtl = (map_env.direction == -1)
    ltr_rx1 = float(SCREEN_WIDTH - map_env.river_x2) if is_rtl else float(map_env.river_x1)
    ltr_rx2 = float(SCREEN_WIDTH - map_env.river_x1) if is_rtl else float(map_env.river_x2)
    ltr_goal_left = float(SCREEN_WIDTH - map_env.goal_rect.right) if is_rtl else float(map_env.goal_rect.left)
    ltr_gate_center = pygame.Vector2(ltr_goal_left, map_env.gate_center.y)
    ltr_bridge_in = pygame.Vector2(ltr_rx1, map_env.bridge_center.y)

    dog_bounds_override: tuple[float, float, float, float] | None = None

    if scenario_mode == "east":
        east_span = max(140.0, ltr_goal_left - ltr_rx2)
        min_x = ltr_rx2 + 28.0
        max_x = ltr_goal_left - 32.0
        min_y, max_y = 90.0, float(SCREEN_HEIGHT - 90)
        spawn_r = 65.0

        if rng.random() < 0.65:
            # Post-bridge exit drill: practice flanking turns right after crossing a shifted bridge
            cx = min(max_x - 75.0, ltr_rx2 + rng.uniform(55.0, 105.0))
            cy = max(115.0, min(float(SCREEN_HEIGHT - 115), float(map_env.bridge_center.y) + rng.uniform(-45.0, 45.0)))
            dog_x = max(ltr_rx1 + 10.0, ltr_rx2 - rng.uniform(0.0, 25.0))
            dog_y = max(75.0, min(float(SCREEN_HEIGHT - 75), float(map_env.bridge_center.y) + rng.uniform(-25.0, 25.0)))
            dog_bounds_override = (
                ltr_rx1 + 5.0,
                max_x,
                max(float(map_env.bridge_y1 + DOG_RADIUS + 4), 65.0),
                min(float(map_env.bridge_y2 - DOG_RADIUS - 4), float(SCREEN_HEIGHT - 65)),
            )
        else:
            cx = rng.uniform(ltr_rx2 + 0.30 * east_span, ltr_rx2 + 0.58 * east_span)
            cy = max(125.0, min(float(SCREEN_HEIGHT - 125), float(ltr_gate_center.y) + rng.uniform(-140.0, 140.0)))
            to_gate = ltr_gate_center - pygame.Vector2(cx, cy)
            push_dir = to_gate.normalize() if to_gate.length_squared() > 1.0 else pygame.Vector2(1.0, 0.0)
            dog_dist = rng.uniform(95.0, 125.0)
            dog_x = max(min_x, min(max_x - 20.0, cx - push_dir.x * dog_dist))
            dog_y = max(75.0, min(float(SCREEN_HEIGHT - 75), cy - push_dir.y * dog_dist + rng.uniform(-25.0, 25.0)))

    elif scenario_mode == "bridge":
        west_edge = max(float(zone_x + 40), ltr_rx1 - 190.0)
        cx = rng.uniform(west_edge, ltr_rx1 - 105.0)
        cy = max(130.0, min(float(SCREEN_HEIGHT - 130), float(map_env.bridge_center.y) + rng.uniform(-120.0, 120.0)))
        spawn_r = 65.0

        min_x = float(zone_x)
        max_x = ltr_rx1 - 30.0
        min_y, max_y = 100.0, float(SCREEN_HEIGHT - 100)

        to_bridge = ltr_bridge_in - pygame.Vector2(cx, cy)
        push_dir = to_bridge.normalize() if to_bridge.length_squared() > 1.0 else pygame.Vector2(1.0, 0.0)
        dog_dist = rng.uniform(110.0, 145.0)
        dog_x = max(60.0, min(max_x - 30.0, cx - push_dir.x * dog_dist))
        dog_y = max(85.0, min(float(SCREEN_HEIGHT - 85), cy - push_dir.y * dog_dist + rng.uniform(-30.0, 30.0)))

    elif scenario_mode == "full_north":
        cx, cy = 260.0, 270.0
        spawn_r = 75.0
        dog_x, dog_y = 110.0, 240.0
        min_x, max_x = float(zone_x + 20), ltr_rx1 - 40.0
        min_y, max_y = 120.0, 680.0

    elif scenario_mode == "full_south":
        cx, cy = 260.0, 530.0
        spawn_r = 75.0
        dog_x, dog_y = 110.0, 560.0
        min_x, max_x = float(zone_x + 20), ltr_rx1 - 40.0
        min_y, max_y = 120.0, 680.0

    elif scenario_mode == "full_center":
        cx, cy = 250.0, 400.0
        spawn_r = 85.0
        dog_x, dog_y = float(DOG_START_POS[0]), float(DOG_START_POS[1])
        min_x, max_x = float(zone_x + 20), ltr_rx1 - 40.0
        min_y, max_y = 120.0, 680.0

    else:  # "full" randomized
        west_limit = max(260.0, ltr_rx1 - 220.0)
        cx = rng.uniform(200.0, west_limit)
        cy = rng.uniform(160.0, float(SCREEN_HEIGHT - 160))
        spawn_r = 80.0
        dog_x = max(55.0, cx - rng.uniform(110.0, 150.0))
        dog_y = max(100.0, min(float(SCREEN_HEIGHT - 100), cy + rng.uniform(-55.0, 55.0)))
        min_x, max_x = float(zone_x), ltr_rx1 - 40.0
        min_y, max_y = 95.0, float(SCREEN_HEIGHT - 95)

    bounds = (min_x, max_x, min_y, max_y)
    raw_coords: list[tuple[float, float]] = []
    for _ in range(count):
        ang = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(8, spawn_r)
        sx = max(min_x, min(max_x, cx + math.cos(ang) * r))
        sy = max(min_y, min(max_y, cy + math.sin(ang) * r))
        raw_coords.append((sx, sy))

    dog_bounds = dog_bounds_override if dog_bounds_override is not None else (
        max(45.0, min_x - 45.0),
        max_x,
        65.0,
        float(SCREEN_HEIGHT - 65),
    )

    if is_rtl:
        cx = float(SCREEN_WIDTH) - cx
        dog_x = float(SCREEN_WIDTH) - dog_x
        bounds = (float(SCREEN_WIDTH) - bounds[1], float(SCREEN_WIDTH) - bounds[0], bounds[2], bounds[3])
        dog_bounds = (
            float(SCREEN_WIDTH) - dog_bounds[1],
            float(SCREEN_WIDTH) - dog_bounds[0],
            dog_bounds[2],
            dog_bounds[3],
        )
        raw_coords = [(float(SCREEN_WIDTH) - sx, sy) for sx, sy in raw_coords]

    herd: list[Sheep] = []
    for sx, sy in raw_coords:
        clear_pos = _place_clear_of_rocks(sx, sy, SHEEP_RADIUS, map_env, bounds=bounds)
        herd.append(Sheep(clear_pos.x, clear_pos.y))

    dog_clear = _place_clear_of_rocks(dog_x, dog_y, DOG_RADIUS, map_env, bounds=dog_bounds)
    dog_angle = math.atan2(cy - dog_clear.y, cx - dog_clear.x)
    return herd, (dog_clear.x, dog_clear.y, dog_angle)


def spawn_training_herd(count: int, map_env: PastureMap, seed: int | None = None) -> list[Sheep]:
    #Spawns sheep grouped in the Western pasture (convenience wrapper).
    herd, _ = spawn_scenario(count, map_env, seed=seed, scenario_mode="full")
    return herd


def evaluate_episode(
    genome: np.ndarray,
    steps: int,
    eval_seed: int,
    sheep_count: int = SHEEP_COUNT,
    scenario_mode: str = "full",
    env_spec: str = "classic",
    env_seed: int | None = None,
) -> dict[str, float | int]:
    #Runs one headless episode on `env_spec` and computes geodesic + herding-stance fitness and telemetry.

    random.seed(eval_seed)
    np.random.seed(eval_seed)

    pasture_map = create_map(env_spec, seed=env_seed if env_seed is not None else eval_seed)
    herd, (dx, dy, d_ang) = spawn_scenario(sheep_count, pasture_map, seed=eval_seed, scenario_mode=scenario_mode)

    dog = HerdingDog(dx, dy)
    dog.reset(dx, dy, facing_angle=d_ang)
    dog.brain.set_flat_genome(genome)

    init_g = [pasture_map.geodesic_distance_to_goal(s.pos, s.radius) for s in herd]
    init_mean_g = sum(init_g) / max(1, len(init_g))
    init_max_g = max(init_g) if init_g else 1.0

    dog_distance_traveled = 0.0
    dog_barks_count = 0
    stance_accum = 0.0
    stance_samples = 0
    time_to_first_cross = -1
    time_to_first_pen = -1
    finish_step = steps
    prev_dog_pos = pygame.Vector2(dog.pos)
    steps_run = 0

    for step in range(steps):
        steps_run = step + 1
        unpenned = [s for s in herd if not s.in_goal_pen]
        if not unpenned:
            finish_step = step
            break

        # Median center over active (unpenned) flock so penned sheep don't pull stragglers into the river
        med_x = statistics.median(s.pos.x for s in unpenned)
        med_y = statistics.median(s.pos.y for s in unpenned)
        flock_center = pygame.Vector2(med_x, med_y)

        dog.perceive_and_act(herd, pasture_map)
        dog_distance_traveled += (dog.pos - prev_dog_pos).length()
        prev_dog_pos = pygame.Vector2(dog.pos)
        if dog.bark_active_time == dog.bark_max_time - 1:
            dog_barks_count += 1

        for s in herd:
            s.update_brain(herd, HERD_MAX_DISTANCE, dog, flock_center, pasture_map)
            s.update_physics(pasture_map)
            if time_to_first_cross == -1 and pasture_map.has_crossed_river(s.pos):
                time_to_first_cross = step
            if time_to_first_pen == -1 and s.in_goal_pen:
                time_to_first_pen = step

        num_s = len(herd)
        for i in range(num_s):
            for j in range(i + 1, num_s):
                herd[i].resolve_collision(herd[j])

        # Continuous herding stance reward sampled every 4th step
        if step % 4 == 0:
            stance_samples += 1
            c_unp = sum((s.pos for s in unpenned), pygame.Vector2(0, 0)) / len(unpenned)
            s_max = max(unpenned, key=lambda s: pasture_map.geodesic_distance_to_goal(s.pos, s.radius))
            focus = s_max.pos if (s_max.pos - c_unp).length() > 140.0 else c_unp
            wp = pasture_map.next_waypoint(focus)
            to_behind = focus - wp
            if to_behind.length_squared() > 0.001:
                ideal_pos = focus + to_behind.normalize() * 70.0
                stance_accum += max(0.0, 1.0 - (dog.pos - ideal_pos).length() / 280.0)

    final_g = [pasture_map.geodesic_distance_to_goal(s.pos, s.radius) for s in herd]
    final_mean_g = sum(final_g) / max(1, len(final_g))
    final_max_g = max(final_g) if final_g else 0.0

    sheep_in_pen = sum(1 for s in herd if pasture_map.is_in_goal_pen(s.pos, s.radius))
    sheep_past_bridge = sum(1 for s in herd if pasture_map.has_crossed_river(s.pos))

    mean_progress = (init_mean_g - final_mean_g) / max(1.0, init_mean_g)
    worst_progress = (init_max_g - final_max_g) / max(1.0, init_max_g)

    progress_score = mean_progress * 1500.0 + worst_progress * 900.0
    bridge_score = (sheep_past_bridge / max(1, sheep_count)) * 600.0
    pen_score = (sheep_in_pen / max(1, sheep_count)) * 3000.0
    stance_score = (stance_accum / max(1, stance_samples)) * 550.0
    completion_bonus = (2500.0 + (steps - finish_step) * 2.5) if sheep_in_pen == sheep_count else 0.0
    bark_penalty = -6.0 * dog_barks_count
    idle_penalty = -300.0 if (dog_distance_traveled / max(1, steps_run)) < 1.0 else 0.0

    total_fitness = (
        progress_score
        + bridge_score
        + pen_score
        + stance_score
        + completion_bonus
        + bark_penalty
        + idle_penalty
    )

    center_all = sum((s.pos for s in herd), pygame.Vector2(0, 0)) / max(1, len(herd))
    flock_spread = sum((s.pos - center_all).length() for s in herd) / max(1, len(herd))

    return {
        "fitness": float(total_fitness),
        "sheep_in_pen": int(sheep_in_pen),
        "corral_ratio": float(sheep_in_pen / max(1, sheep_count)),
        "sheep_past_bridge": int(sheep_past_bridge),
        "bridge_ratio": float(sheep_past_bridge / max(1, sheep_count)),
        "flock_spread": float(flock_spread),
        "dog_distance_traveled": float(dog_distance_traveled),
        "dog_barks_count": int(dog_barks_count),
        "time_to_first_cross": int(time_to_first_cross),
        "time_to_first_pen": int(time_to_first_pen),
    }


def evaluate_genome_batch(
    item: tuple[np.ndarray, list[tuple], int]
) -> dict[str, float | int]:
    """
    Worker function: evaluates one genome across multiple scenarios/environments and averages metrics.
    Each episode tuple may be either:
      - (scenario_mode, steps, seed)
      - (scenario_mode, steps, seed, env_spec, env_seed)
    """
    genome, episodes, sheep_count = item
    ep_results = []
    for ep in episodes:
        if len(ep) == 3:
            mode, steps, seed = ep
            env_spec, env_seed = "classic", None
        else:
            mode, steps, seed, env_spec, env_seed = ep
        ep_results.append(
            evaluate_episode(
                genome,
                steps,
                seed,
                sheep_count=sheep_count,
                scenario_mode=mode,
                env_spec=env_spec,
                env_seed=env_seed,
            )
        )

    n = max(1, len(ep_results))
    fits = [r["fitness"] for r in ep_results]
    mean_fit = sum(fits) / n
    combined_fit = (0.75 * mean_fit + 0.25 * min(fits)) if n > 1 else mean_fit
    return {
        "fitness": combined_fit,
        "corral_ratio": sum(r["corral_ratio"] for r in ep_results) / n,
        "bridge_ratio": sum(r["bridge_ratio"] for r in ep_results) / n,
        "sheep_in_pen": int(round(sum(r["sheep_in_pen"] for r in ep_results) / n)),
        "sheep_past_bridge": int(round(sum(r["sheep_past_bridge"] for r in ep_results) / n)),
        "flock_spread": sum(r["flock_spread"] for r in ep_results) / n,
        "dog_distance_traveled": sum(r["dog_distance_traveled"] for r in ep_results) / n,
        "dog_barks_count": int(round(sum(r["dog_barks_count"] for r in ep_results) / n)),
    }


class GeneticTrainer:
    """
    Rank-Based Antithetic Evolution Strategy (OpenAI-ES with Adam + Elite Archive).
    Supports 2-stage curriculum learning, domain-randomized multi-environment training (`multi`),
    persistent worker pooling, and fixed multi-map validation.
    """

    def __init__(
        self,
        population_size: int = GA_POPULATION_SIZE,
        model_path: str = MODEL_SAVE_PATH,
        num_workers: int = 1,
        seed: int | None = None,
        total_generations: int = 60,
        sheep_count: int = SHEEP_COUNT,
        seed_from_disk: bool = True,
        env_mode: str = "multi",
    ):
        self.pop_size = max(4, population_size)
        self.model_path = model_path
        self.num_workers = max(1, num_workers)
        self.total_generations = max(1, total_generations)
        self.sheep_count = sheep_count
        self.base_seed = seed if seed is not None else 42
        self.env_mode = resolve_env_id(env_mode)

        template_brain = HerdingTransformerBrain()
        self.mean_genome = template_brain.get_flat_genome().astype(np.float32)
        self.genome_size = self.mean_genome.shape[0]

        if seed_from_disk and os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location="cpu")
                template_brain.load_state_dict(state_dict)
                self.mean_genome = template_brain.get_flat_genome().astype(np.float32)
            except Exception:
                pass

        # Adam state
        self.m = np.zeros(self.genome_size, dtype=np.float32)
        self.v = np.zeros(self.genome_size, dtype=np.float32)
        self.adam_t = 0

        self.best_overall_fitness = -1e9
        self.best_overall_genome: np.ndarray | None = None
        self.generation = 0
        self.stage1_gens = int(self.total_generations * 0.30)

        self._pool: multiprocessing.pool.Pool | None = None
        if self.num_workers > 1:
            self._pool = multiprocessing.Pool(self.num_workers, initializer=_init_worker)

    def close(self):
        #Closes persistent worker pool cleanly.
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.close()

    def _get_episodes(
        self, gen: int, steps_per_eval: int, fixed_seed: int | None = None
    ) -> list[tuple[str, int, int, str, int]]:
        s_base = fixed_seed if fixed_seed is not None else (self.base_seed * 1000 + gen * 17)

        if self.env_mode == "multi":
            if gen <= self.stage1_gens:
                east_spec = "variant:north_south" if gen % 2 == 1 else "variant:south_north"
                return [
                    ("east", min(1050, steps_per_eval), s_base + 1, east_spec, s_base + 101),
                    ("bridge", min(1100, steps_per_eval), s_base + 2, "random", s_base + 102),
                    ("full", steps_per_eval, s_base + 3, "random", s_base + 103),
                ]
            if gen % 3 == 0:
                specs = SELECTABLE_ENV_IDS
            elif gen % 3 == 1:
                specs = ("variant:classic", "variant:north_south", "variant:south_north")
            else:
                specs = ("random", "variant:north_south", "variant:south_north")
            return [
                ("full", steps_per_eval, s_base + 1, specs[0], s_base + 201),
                ("full", steps_per_eval, s_base + 2, specs[1], s_base + 202),
                ("full", steps_per_eval, s_base + 3, specs[2], s_base + 203),
            ]

        single_spec = self.env_mode if self.env_mode != "random" else "random"
        if gen <= self.stage1_gens:
            return [
                ("east", min(900, steps_per_eval), s_base + 1, single_spec, s_base + 101),
                ("bridge", min(1100, steps_per_eval), s_base + 2, single_spec, s_base + 102),
                ("full", steps_per_eval, s_base + 3, single_spec, s_base + 103),
            ]
        return [
            ("full", steps_per_eval, s_base + 1, single_spec, s_base + 201),
            ("full", steps_per_eval, s_base + 2, single_spec, s_base + 202),
            ("full", steps_per_eval, s_base + 3, single_spec, s_base + 203),
        ]

    def _validate_full_map(self, genome: np.ndarray, steps_per_eval: int) -> dict[str, float | int]:
        #Evaluates a candidate on a fixed multi-map or single-map validation suite.
        if self.env_mode == "multi":
            val_episodes = [
                ("full", steps_per_eval, 60001, "classic", 60001),
                ("full", steps_per_eval, 60002, "north_south", 60002),
                ("full", steps_per_eval, 60003, "south_north", 60003),
                ("full", steps_per_eval, 60004, "random", 60004),
                ("full", steps_per_eval, 60005, "random", 60005),
                ("full", steps_per_eval, 60006, "random", 60006),
            ]
        else:
            val_seeds = [50001, 50002, 50003, 50004]
            val_episodes = [("full", steps_per_eval, s, self.env_mode, s) for s in val_seeds]

        if self._pool is not None:
            items = [(genome, [ep], self.sheep_count) for ep in val_episodes]
            ep_results = self._pool.map(evaluate_genome_batch, items, chunksize=1)
            n = len(ep_results)
            fits = [r["fitness"] for r in ep_results]
            mean_fit = sum(fits) / n
            score = (0.75 * mean_fit + 0.25 * min(fits)) if n > 1 else mean_fit
            return {
                "fitness": score,
                "corral_ratio": sum(r["corral_ratio"] for r in ep_results) / n,
                "bridge_ratio": sum(r["bridge_ratio"] for r in ep_results) / n,
            }
        return evaluate_genome_batch((genome, val_episodes, self.sheep_count))

    def step_generation(
        self, steps_per_eval: int = GA_SIMULATION_STEPS, fixed_seed: int | None = None
    ) -> dict:
        #Runs one Evolution Strategy generation (curriculum + antithetic perturbations + Adam).
        self.generation += 1
        progress = min(1.0, (self.generation - 1) / max(1, self.total_generations - 1))
        sigma = ES_SIGMA_START + (ES_SIGMA_END - ES_SIGMA_START) * progress
        lr = ES_LR_START + (ES_LR_END - ES_LR_START) * progress

        episodes = self._get_episodes(self.generation, steps_per_eval, fixed_seed=fixed_seed)

        # Build antithetic population + mean + incumbent elite
        num_pairs = max(1, (self.pop_size - 2) // 2)
        epsilons = np.random.randn(num_pairs, self.genome_size).astype(np.float32)

        candidates: list[np.ndarray] = []
        for k in range(num_pairs):
            candidates.append(self.mean_genome + sigma * epsilons[k])
            candidates.append(self.mean_genome - sigma * epsilons[k])

        candidates.append(self.mean_genome.copy())
        candidates.append(
            self.best_overall_genome.copy() if self.best_overall_genome is not None else self.mean_genome.copy()
        )

        work_items = [(cand, episodes, self.sheep_count) for cand in candidates]
        if self._pool is not None:
            results = self._pool.map(evaluate_genome_batch, work_items, chunksize=1)
        else:
            results = [evaluate_genome_batch(item) for item in work_items]

        # Centered rank transformation across the 2*num_pairs antithetic evaluations
        es_count = 2 * num_pairs
        es_fits = np.array([results[i]["fitness"] for i in range(es_count)], dtype=np.float32)
        ranks = np.empty(es_count, dtype=np.float32)
        ranks[np.argsort(es_fits)] = np.arange(es_count, dtype=np.float32)
        centered_ranks = (ranks / max(1, es_count - 1)) - 0.5

        pos_ranks = centered_ranks[0::2]
        neg_ranks = centered_ranks[1::2]
        rank_diffs = pos_ranks - neg_ranks  # Shape: [num_pairs]

        grad_raw = np.sum(
            rank_diffs.astype(np.float64)[:, None] * epsilons.astype(np.float64), axis=0
        ).astype(np.float32)
        grad = grad_raw / (num_pairs * sigma) - ES_WEIGHT_DECAY * self.mean_genome

        # Adam ascent step
        self.adam_t += 1
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        self.m = beta1 * self.m + (1.0 - beta1) * grad
        self.v = beta2 * self.v + (1.0 - beta2) * (grad * grad)
        m_hat = self.m / (1.0 - beta1 ** self.adam_t)
        v_hat = self.v / (1.0 - beta2 ** self.adam_t)
        self.mean_genome = self.mean_genome + lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Find best candidate this generation
        best_idx = int(np.argmax([r["fitness"] for r in results]))
        best_metrics = results[best_idx]
        best_cand = candidates[best_idx]

        all_fits = [r["fitness"] for r in results]
        mean_fit = float(statistics.mean(all_fits))
        std_fit = float(statistics.stdev(all_fits)) if len(all_fits) > 1 else 0.0

        # Validate on full-map fixed benchmark at Stage 1 transition, during Stage 2, or final gen
        is_stage2 = self.generation > self.stage1_gens
        is_transition = self.generation == self.stage1_gens
        is_final = self.generation == self.total_generations
        saved_new_best = False

        if is_stage2 or is_transition or is_final:
            probe = best_cand if is_stage2 else self.mean_genome
            if is_transition or is_final or best_metrics["fitness"] > self.best_overall_fitness * 0.95:
                val_res = self._validate_full_map(probe, steps_per_eval)
                if val_res["fitness"] > self.best_overall_fitness:
                    self.best_overall_fitness = float(val_res["fitness"])
                    self.best_overall_genome = probe.copy()
                    self.save_model(self.best_overall_genome)
                    saved_new_best = True

        return {
            "generation": self.generation,
            "stage": 2 if is_stage2 else 1,
            "best_fitness": float(best_metrics["fitness"]),
            "mean_fitness": mean_fit,
            "std_fitness": std_fit,
            "best_corral_ratio": float(best_metrics["corral_ratio"]),
            "best_penned": int(best_metrics["sheep_in_pen"]),
            "best_bridge_ratio": float(best_metrics["bridge_ratio"]),
            "best_past_bridge": int(best_metrics["sheep_past_bridge"]),
            "flock_spread": float(best_metrics["flock_spread"]),
            "dog_path_length": float(best_metrics["dog_distance_traveled"]),
            "dog_barks": int(best_metrics["dog_barks_count"]),
            "overall_best": float(self.best_overall_fitness),
            "saved_new_best": saved_new_best,
        }

    def save_model(self, genome: np.ndarray | None = None, filepath: str | None = None):
        target_path = filepath if filepath is not None else self.model_path
        target_genome = genome if genome is not None else self.best_overall_genome

        if target_genome is None:
            return

        brain = HerdingTransformerBrain()
        brain.set_flat_genome(target_genome)
        torch.save(brain.state_dict(), target_path)

    def load_model(self, filepath: str | None = None) -> HerdingTransformerBrain:
        target_path = filepath if filepath is not None else self.model_path
        brain = HerdingTransformerBrain()
        if os.path.exists(target_path):
            state_dict = torch.load(target_path, map_location="cpu")
            brain.load_state_dict(state_dict)
        return brain
