"""
Main entry point for the Autonomous Transformer Herding Dog Simulation.
Supports:
1. Visual Simulation Mode (default): showcasing the autonomous transformer dog,
   its senses (vision rays, hearing, smell), river, bridge, rocks, and sheep flocking
   into the goal pen.
2. Training Mode (--mode train): Headless Evolution Strategy neuroevolution
   across domain-randomized maps (`--env multi` default) or a single preset.
"""

from __future__ import annotations
import argparse
import multiprocessing
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import random
import statistics
import sys
import time
import pygame
import torch
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FPS,
    TITLE,
    COLOR_GRASS,
    COLOR_GRASS_DARK,
    COLOR_TEXT,
    COLOR_TEXT_SHADOW,
    SHEEP_COUNT,
    HERD_MAX_DISTANCE,
    MODEL_SAVE_PATH,
    GA_POPULATION_SIZE,
    GA_MAX_GENERATIONS,
    GA_SIMULATION_STEPS,
)
from map_env import SELECTABLE_ENV_IDS, create_map, resolve_env_id
from transformer_brain import HerdingTransformerBrain
from dog import HerdingDog
from sheep import Sheep
from genetic_algorithm import GeneticTrainer, spawn_scenario


def generate_grass_patches(width: int, height: int, count: int = 90) -> list[tuple[int, int, int]]:
    # Generates random coordinates for decorative grass tufts.
    random.seed(42)
    patches = []
    for _ in range(count):
        x = random.randint(25, width - 25)
        y = random.randint(25, height - 25)
        length = random.randint(4, 9)
        patches.append((x, y, length))
    return patches


def draw_grass_background(surface: pygame.Surface, patches: list[tuple[int, int, int]]):
    # Draws rich pasture grass background with blade details.
    surface.fill(COLOR_GRASS)
    for x, y, length in patches:
        pygame.draw.line(surface, COLOR_GRASS_DARK, (x, y), (x - 2, y - length), 1)
        pygame.draw.line(surface, COLOR_GRASS_DARK, (x + 2, y), (x + 3, y - length + 1), 1)


def draw_shadowed_text(surface: pygame.Surface, text: str, font: pygame.font.Font, x: int, y: int, color: tuple = COLOR_TEXT):
    # Helper to draw text with drop shadow for optimal contrast.
    shadow_surf = font.render(text, True, COLOR_TEXT_SHADOW)
    text_surf = font.render(text, True, color)
    surface.blit(shadow_surf, (x + 1, y + 1))
    surface.blit(text_surf, (x, y))


def run_training_mode(args):
    # Executes headless Evolution Strategy neuroevolution to train the transformer dog.
    env_mode = resolve_env_id(args.env) if args.env else "multi"
    steps_per_eval = args.steps if args.steps is not None else (1500 if env_mode == "multi" else GA_SIMULATION_STEPS)
    env_desc = (
        "Multi-Environment Domain Randomization (shifted bridges, moved goals, modified rocks + 3 presets)"
        if env_mode == "multi"
        else f"Single Environment ({env_mode})"
    )

    print("=" * 72)
    print(" AUTONOMOUS TRANSFORMER HERDING DOG - ES CURRICULUM TRAINING ")
    print("=" * 72)
    print(f"Environment Mode:      {env_desc}")
    print(f"Population Size:       {args.pop_size}")
    print(f"Max Generations:       {args.generations}")
    print(f"Simulation Steps/Eval: {steps_per_eval} frames (~{steps_per_eval/60:.1f}s physics)")
    print(f"Parallel CPU Workers:  {args.workers}")
    print(f"Target Model File:     '{args.model}' ({'fresh init' if args.fresh else 'warm-start if exists'})")
    print("-" * 72)

    trainer = GeneticTrainer(
        population_size=args.pop_size,
        model_path=args.model,
        num_workers=args.workers,
        total_generations=args.generations,
        sheep_count=args.sheep,
        seed_from_disk=not args.fresh,
        env_mode=env_mode,
    )

    start_time = time.time()
    try:
        for gen in range(1, args.generations + 1):
            gen_start = time.time()
            res = trainer.step_generation(steps_per_eval=steps_per_eval)
            gen_elapsed = time.time() - gen_start

            best_fit = res["best_fitness"]
            mean_fit = res["mean_fitness"]
            penned = res["best_penned"]
            crossed = res["best_past_bridge"]

            status = (
                f"Gen {gen:03d}/{args.generations} [S{res['stage']}] | "
                f"Best: {best_fit:7.1f} | Mean: {mean_fit:7.1f} | "
                f"Penned: {penned:02d}/{args.sheep} | Crossed: {crossed:02d} | Time: {gen_elapsed:4.1f}s"
            )
            if res["saved_new_best"]:
                status += "  [NEW BEST! Saved]"
            print(status)
    finally:
        trainer.close()

    total_time = time.time() - start_time
    print("=" * 72)
    print(f"Training completed in {total_time:.1f} seconds.")
    print(f"Best overall model saved to '{args.model}'.")
    print("Run visual simulation with: python3 main.py --mode play")
    print("=" * 72)


def run_simulation_mode(args):
    # Executes real-time visual simulation with Pygame (switch maps with keys 1, 2, 3 or --env).
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Helvetica, Arial, sans-serif", 16, bold=True)
    small_font = pygame.font.SysFont("Helvetica, Arial, sans-serif", 13)
    large_font = pygame.font.SysFont("Helvetica, Arial, sans-serif", 28, bold=True)

    grass_patches = generate_grass_patches(SCREEN_WIDTH, SCREEN_HEIGHT)

    if args.env is None:
        initial_env_spec = SELECTABLE_ENV_IDS[0]
    else:
        resolved = resolve_env_id(args.env)
        initial_env_spec = "random" if resolved == "multi" else resolved

    pasture_map = create_map(initial_env_spec, seed=int(time.time() * 1000) % 100000)

    brain = HerdingTransformerBrain()
    model_loaded = False
    if os.path.exists(args.model):
        try:
            state_dict = torch.load(args.model, map_location="cpu")
            brain.load_state_dict(state_dict)
            model_loaded = True
            print(f"[Simulation] Loaded trained Transformer model from '{args.model}'.")
        except Exception as e:
            print(f"[Simulation] Warning: Could not load '{args.model}' ({e}). Using random brain.")
    else:
        print(f"[Simulation] Note: '{args.model}' not found. Using untrained brain. (Train with --mode train)")

    def load_environment(env_spec: str):
        nonlocal pasture_map, herd, dog, is_paused
        seed_val = int(time.time() * 1000) % 1000000
        pasture_map = create_map(env_spec, seed=seed_val)
        herd, (nx, ny, n_ang) = spawn_scenario(args.sheep, pasture_map, seed=seed_val, scenario_mode="full")
        dog.reset(nx, ny, facing_angle=n_ang)
        is_paused = False
        print(f"[Simulation] Switched to environment: {pasture_map.name}")

    herd, (dx, dy, d_ang) = spawn_scenario(args.sheep, pasture_map, seed=int(time.time()), scenario_mode="full")
    dog = HerdingDog(dx, dy, brain=brain)
    dog.reset(dx, dy, facing_angle=d_ang)

    show_senses = not args.no_senses
    show_debug = args.debug
    is_paused = False

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_1:
                    load_environment(SELECTABLE_ENV_IDS[0])
                elif event.key == pygame.K_2:
                    load_environment(SELECTABLE_ENV_IDS[1])
                elif event.key == pygame.K_3:
                    load_environment(SELECTABLE_ENV_IDS[2])
                elif event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                elif event.key == pygame.K_s:
                    show_senses = not show_senses
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif event.key == pygame.K_b:
                    dog.bark()
                elif event.key == pygame.K_r:
                    herd, (rx, ry, r_ang) = spawn_scenario(
                        args.sheep, pasture_map, seed=int(time.time() * 1000) % 100000, scenario_mode="full"
                    )
                    dog.reset(rx, ry, facing_angle=r_ang)
                elif event.key == pygame.K_t:
                    print("[Interactive] Running 3 quick multi-environment ES generations...")
                    trainer = GeneticTrainer(
                        population_size=14,
                        model_path=args.model,
                        num_workers=args.workers,
                        total_generations=3,
                        sheep_count=args.sheep,
                        env_mode="multi",
                    )
                    try:
                        for _ in range(3):
                            trainer.step_generation(steps_per_eval=600)
                    finally:
                        trainer.close()
                    if os.path.exists(args.model):
                        brain.load_state_dict(torch.load(args.model, map_location="cpu"))
                        model_loaded = True
                        print("[Interactive] Updated dog with newly evolved weights!")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:
                    dog.bark()
                elif event.button == 3:
                    herd.append(Sheep(mx, my))

        if not is_paused:
            unpenned = [s for s in herd if not s.in_goal_pen]
            active = unpenned if unpenned else herd
            med_x = statistics.median(s.pos.x for s in active)
            med_y = statistics.median(s.pos.y for s in active)
            flock_center = pygame.Vector2(med_x, med_y)

            dog.perceive_and_act(herd, pasture_map)

            for sheep in herd:
                sheep.update_brain(
                    herd,
                    herd_max_dist=HERD_MAX_DISTANCE,
                    dog=dog,
                    flock_center=flock_center,
                    pasture_map=pasture_map,
                )
                sheep.update_physics(pasture_map=pasture_map)

            num_sheep = len(herd)
            for _ in range(3):
                for i in range(num_sheep):
                    for j in range(i + 1, num_sheep):
                        herd[i].resolve_collision(herd[j])

        draw_grass_background(screen, grass_patches)
        pasture_map.draw(screen)

        penned_count = 0
        past_bridge_count = 0
        for sheep in herd:
            if pasture_map.is_in_goal_pen(sheep.pos, sheep.radius):
                penned_count += 1
            if sheep.pos.x > pasture_map.river_x2:
                past_bridge_count += 1
            sheep.draw(screen, show_debug=show_debug, herd_max_dist=HERD_MAX_DISTANCE)

        dog.draw(screen, show_senses=show_senses, show_debug=show_debug)

        env_tag = (
            f"[{SELECTABLE_ENV_IDS.index(pasture_map.env_id) + 1}] {pasture_map.name}"
            if pasture_map.env_id in SELECTABLE_ENV_IDS
            else pasture_map.name
        )
        hud_top = (
            f"FPS: {int(clock.get_fps())} | Corralled: {penned_count}/{len(herd)} | "
            f"Bridge: {past_bridge_count}/{len(herd)} | Env: {env_tag}"
        )
        draw_shadowed_text(screen, hud_top, font, 16, 12)

        dog_speed = dog.vel.length()
        dog_action_str = (
            f"Brain: {'Trained' if model_loaded else 'Initial'} | Cmd=({dog.last_ax:+.2f},{dog.last_ay:+.2f}) | "
            f"Speed={dog_speed:.1f}px/f | Bark={'ACTIVE' if dog.bark_active_time>0 else 'READY'}"
        )
        draw_shadowed_text(screen, dog_action_str, small_font, 16, 36)

        senses_str = (
            f"Senses [S]: Vision=9 rays | Hearing Behind={dog.hearing_behind_intensity*100:.0f}% | "
            f"Smell=Herd/Straggler/Waypoint"
        )
        draw_shadowed_text(screen, senses_str, small_font, 16, 56, color=(240, 240, 180))

        controls_text = (
            "Controls: [1/2/3] Switch Map | [Space] Pause | "
            "[S] Senses | [D] Debug | [R] Respawn | [Q] Quit"
        )
        draw_shadowed_text(screen, controls_text, small_font, 16, SCREEN_HEIGHT - 24)

        if penned_count == len(herd) and len(herd) > 0:
            banner_surf = large_font.render("HERD SUCCESSFULLY CORRALLED!", True, (255, 240, 80))
            shadow_banner = large_font.render("HERD SUCCESSFULLY CORRALLED!", True, (20, 40, 20))
            bx = SCREEN_WIDTH // 2 - banner_surf.get_width() // 2
            by = SCREEN_HEIGHT // 2 - 30
            screen.blit(shadow_banner, (bx + 2, by + 2))
            screen.blit(banner_surf, (bx, by))

        if is_paused:
            pause_surf = large_font.render("-- PAUSED --", True, (255, 220, 100))
            screen.blit(pause_surf, (SCREEN_WIDTH // 2 - pause_surf.get_width() // 2, 70))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


def main():
    parser = argparse.ArgumentParser(description="Autonomous Transformer Herding Dog Simulation & ES Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="play",
        choices=["play", "eval", "train"],
        help="Mode: 'play' / 'eval' for visual simulation, 'train' for headless ES neuroevolution",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["1", "2", "3", "classic", "north_south", "south_north", "random", "multi"],
        help=(
            "Environment selection: 1/classic, 2/north_south, 3/south_north, random (procedural variant), "
            "or multi (domain randomization for training; defaults to 1/classic in play mode)"
        ),
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=GA_MAX_GENERATIONS,
        help=f"Number of generations for training (default: {GA_MAX_GENERATIONS})",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=GA_POPULATION_SIZE,
        help=f"Population size for ES (default: {GA_POPULATION_SIZE})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Simulation steps per evaluation (default: 1500 for multi-env, {GA_SIMULATION_STEPS} for single-env)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start training from fresh random weights instead of loading --model from disk",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 2),
        help=f"Number of parallel CPU worker processes for training (default: {max(1, multiprocessing.cpu_count() - 2)})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_SAVE_PATH,
        help="Path to save or load the transformer genome weights (default: best_dog_genome.pt)",
    )
    parser.add_argument(
        "--sheep",
        type=int,
        default=SHEEP_COUNT,
        help=f"Number of sheep in herd (default: {SHEEP_COUNT})",
    )
    parser.add_argument(
        "--no-senses",
        action="store_true",
        help="Hide sensory visualization rays by default",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug collision circles and state badges",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training_mode(args)
    else:
        run_simulation_mode(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
