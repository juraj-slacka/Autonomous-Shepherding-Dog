"""
Executes N = 30 independent evolutionary training runs across randomized seeds

Usage
  30-run full execution:
    python3 run_30_experiments.py

  Quick end-to-end smoke test:
    python3 run_30_experiments.py --dry-run --workers 4 --output-dir /tmp/test_30
"""

from __future__ import annotations
import argparse
import json
import multiprocessing
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import random
import time
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import torch

from config import (
    GA_POPULATION_SIZE,
    SHEEP_COUNT,
)
from map_env import SELECTABLE_ENV_IDS, resolve_env_id
from transformer_brain import HerdingTransformerBrain
from genetic_algorithm import (
    GeneticTrainer,
    _init_worker,
    evaluate_genome_batch,
)


ALL_EVAL_ENVS = ["classic", "north_south", "south_north", "random"]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 12345) -> tuple[float, float]:
    # Computes percentile 95% bootstrap confidence interval for the sample mean.
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1 or np.ptp(arr) == 0.0:
        m = float(np.mean(arr)) if len(arr) else 0.0
        return m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def safe_shapiro(values: np.ndarray) -> tuple[float, float, str]:
    # Runs Shapiro-Wilk normality check guarding against small/constant vectors.
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 3 or np.ptp(arr) < 1e-12:
        return float("nan"), 1.0, "constant_or_insufficient_n"
    stat, p = scipy_stats.shapiro(arr)
    return float(stat), float(p), "ok"


def safe_mannwhitney(a: np.ndarray, b: np.ndarray) -> dict:
    # Runs two-sided Mann-Whitney U test with rank-biserial correlation effect size.
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    u_stat, p_val = scipy_stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    n1, n2 = len(arr_a), len(arr_b)
    r_rb = (2.0 * float(u_stat) / max(1, n1 * n2)) - 1.0
    return {"U": float(u_stat), "p_value": float(p_val), "rank_biserial_r": float(r_rb)}


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    # Runs paired Wilcoxon signed-rank test guarding against all-zero differences.
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if np.allclose(diff, 0.0) or len(diff) < 1:
        return {"stat": 0.0, "p_value": 1.0}
    stat, p_val = scipy_stats.wilcoxon(a, b, zero_method="wilcox")
    return {"stat": float(stat), "p_value": float(p_val)}


def evaluate_policy_on_suite(
    genome: np.ndarray,
    eval_trials: int,
    sheep_count: int,
    horizons: tuple[int, ...] = (1200, 1500),
    num_workers: int = 4,
) -> dict:
    """
    Evaluates one genome across fixed paired scenarios so all 30 trained policies
    and all 30 untrained baseline policies face identical unseen maps and spawns.
    """
    jobs = []
    meta = []
    for steps in horizons:
        for env_spec in ALL_EVAL_ENVS:
            for t in range(eval_trials):
                eval_seed = 80000 + t * 17
                env_seed = 85000 + t * 17
                jobs.append((genome, [("full", steps, eval_seed, env_spec, env_seed)], sheep_count))
                meta.append((steps, env_spec, t))

    if num_workers > 1:
        with multiprocessing.Pool(num_workers, initializer=_init_worker) as pool:
            results = pool.map(evaluate_genome_batch, jobs, chunksize=1)
    else:
        results = [evaluate_genome_batch(j) for j in jobs]

    summary: dict = {}
    for steps in horizons:
        key_h = f"steps_{steps}"
        summary[key_h] = {}
        all_corral, all_bridge = [], []
        for env_spec in ALL_EVAL_ENVS:
            sub = [r for m, r in zip(meta, results) if m[0] == steps and m[1] == env_spec]
            c_pct = [r["corral_ratio"] * 100.0 for r in sub]
            b_pct = [r["bridge_ratio"] * 100.0 for r in sub]
            sp = [r["flock_spread"] for r in sub]
            full_cnt = sum(1 for r in sub if int(r["sheep_in_pen"]) == sheep_count)
            summary[key_h][env_spec] = {
                "corral_mean": float(np.mean(c_pct)),
                "bridge_mean": float(np.mean(b_pct)),
                "spread_mean": float(np.mean(sp)),
                "full_corral_trials": int(full_cnt),
                "total_trials": len(sub),
            }
            all_corral.extend(c_pct)
            all_bridge.extend(b_pct)
        summary[key_h]["overall_corral_mean"] = float(np.mean(all_corral))
        summary[key_h]["overall_bridge_mean"] = float(np.mean(all_bridge))
    return summary


def run_single_experiment(
    run_idx: int,
    seed: int,
    args,
    env_mode: str,
    run_dir: str,
) -> tuple[pd.DataFrame, dict]:
    # Executes one complete independent seed (or loads it cleanly on resume).
    os.makedirs(run_dir, exist_ok=True)
    done_flag = os.path.join(run_dir, "DONE")
    metrics_path = os.path.join(run_dir, "training_metrics.csv")
    model_path = os.path.join(run_dir, "best_model.pt")
    eval_path = os.path.join(run_dir, "eval_results.json")

    if (
        os.path.exists(done_flag)
        and os.path.exists(metrics_path)
        and os.path.exists(model_path)
        and os.path.exists(eval_path)
    ):
        print(f"[Resume] Run {run_idx:02d}/{args.num_runs} (Seed {seed}) already complete; loading cached results.")
        df_seed = pd.read_csv(metrics_path)
        with open(eval_path) as f:
            eval_data = json.load(f)
        return df_seed, eval_data

    print(f"\n>>> Starting Run {run_idx:02d}/{args.num_runs} [Seed {seed} | Env: {env_mode} | Gens: {args.generations}] <<<")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = GeneticTrainer(
        population_size=args.pop_size,
        model_path=model_path,
        num_workers=args.workers,
        seed=seed,
        total_generations=args.generations,
        sheep_count=args.sheep,
        seed_from_disk=False,
        env_mode=env_mode,
    )

    records = []
    t_run_start = time.time()
    try:
        for gen in range(1, args.generations + 1):
            t0 = time.time()
            res = trainer.step_generation(steps_per_eval=args.steps)
            dt = time.time() - t0
            records.append(
                {
                    "run_idx": run_idx,
                    "seed": seed,
                    "generation": gen,
                    "stage": res["stage"],
                    "best_fitness": res["best_fitness"],
                    "mean_fitness": res["mean_fitness"],
                    "std_fitness": res["std_fitness"],
                    "best_corral_ratio": res["best_corral_ratio"],
                    "best_penned": res["best_penned"],
                    "best_bridge_ratio": res["best_bridge_ratio"],
                    "best_past_bridge": res["best_past_bridge"],
                    "flock_spread": res["flock_spread"],
                    "dog_path_length": res["dog_path_length"],
                    "dog_barks": res["dog_barks"],
                    "duration_sec": dt,
                }
            )
            if gen == 1 or gen == trainer.stage1_gens or gen == trainer.stage1_gens + 1 or gen % 10 == 0 or gen == args.generations:
                star = " *" if res["saved_new_best"] else ""
                print(
                    f"  [Run {run_idx:02d} | Seed {seed} | S{res['stage']}] Gen {gen:3d}/{args.generations} | "
                    f"Best: {res['best_fitness']:7.1f} | Mean: {res['mean_fitness']:7.1f} | "
                    f"Bridge: {res['best_bridge_ratio']*100:5.1f}% | Corral: {res['best_corral_ratio']*100:5.1f}%{star}"
                )

        if trainer.best_overall_genome is None:
            trainer.save_model(trainer.mean_genome.copy(), model_path)
            best_genome = trainer.mean_genome.copy()
        else:
            best_genome = trainer.best_overall_genome.copy()
        val_fit = float(trainer.best_overall_fitness)
    finally:
        trainer.close()

    df_seed = pd.DataFrame(records)
    df_seed.to_csv(metrics_path, index=False)

    horizons = (1200, args.steps) if args.steps != 1200 else (1200,)
    eval_suite = evaluate_policy_on_suite(
        best_genome, eval_trials=args.eval_trials, sheep_count=args.sheep, horizons=horizons, num_workers=args.workers
    )
    train_elapsed_min = (time.time() - t_run_start) / 60.0

    last_row = df_seed.iloc[-1]
    eval_data = {
        "run_idx": run_idx,
        "seed": seed,
        "train_minutes": float(train_elapsed_min),
        "val_fitness": float(val_fit),
        "final_best_fitness": float(last_row["best_fitness"]),
        "final_mean_fitness": float(last_row["mean_fitness"]),
        "final_bridge_pct": float(last_row["best_bridge_ratio"] * 100.0),
        "final_corral_pct": float(last_row["best_corral_ratio"] * 100.0),
        "eval": eval_suite,
    }
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    with open(done_flag, "w") as f:
        f.write(f"completed {datetime.now().isoformat()}\n")

    primary_h = f"steps_{args.steps}"
    s_h = eval_suite[primary_h]
    print(
        f"  -> Run {run_idx:02d} done in {train_elapsed_min:.1f} min | "
        f"Env1={s_h['classic']['corral_mean']:.1f}% | "
        f"Env2={s_h['north_south']['corral_mean']:.1f}% | "
        f"Env3={s_h['south_north']['corral_mean']:.1f}% | "
        f"Random={s_h['random']['corral_mean']:.1f}%"
    )
    return df_seed, eval_data


def get_or_compute_untrained_baseline(args, output_dir: str) -> list[dict]:
    # Evaluates N untrained random policies matching `--num-runs` on the exact same test suite (cached).
    cache_path = os.path.join(output_dir, "untrained_baseline.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) >= args.num_runs and cached[0].get("steps") == args.steps and cached[0].get("eval_trials") == args.eval_trials:
            return cached[: args.num_runs]

    print(f"\n[Baseline] Evaluating {args.num_runs} independent untrained Transformer policies...")
    horizons = (1200, args.steps) if args.steps != 1200 else (1200,)
    baseline_list = []
    for idx in range(args.num_runs):
        b_seed = 50000 + idx
        torch.manual_seed(b_seed)
        np.random.seed(b_seed)
        rand_genome = HerdingTransformerBrain().get_flat_genome()
        res = evaluate_policy_on_suite(
            rand_genome, eval_trials=args.eval_trials, sheep_count=args.sheep, horizons=horizons, num_workers=args.workers
        )
        baseline_list.append({"idx": idx, "seed": b_seed, "steps": args.steps, "eval_trials": args.eval_trials, "eval": res})

    with open(cache_path, "w") as f:
        json.dump(baseline_list, f, indent=2)
    return baseline_list


def generate_statistical_outputs(
    all_dfs: list[pd.DataFrame],
    all_evals: list[dict],
    untrained_list: list[dict],
    args,
    output_dir: str,
):
    # Computes full descriptive + nonparametric statistics, CSV/JSON summaries and 300-DPI figures.
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(os.path.join(output_dir, "all_training_metrics.csv"), index=False)

    primary_key = f"steps_{args.steps}"
    rows = []
    for ev in all_evals:
        s_p = ev["eval"][primary_key]
        row = {
            "run_idx": ev["run_idx"],
            "seed": ev["seed"],
            "train_minutes": ev["train_minutes"],
            "val_fitness": ev["val_fitness"],
            "final_best_fitness": ev["final_best_fitness"],
            "final_mean_fitness": ev["final_mean_fitness"],
            "final_corral_pct": ev["final_corral_pct"],
            "overall_corral_pct": s_p["overall_corral_mean"],
            "overall_bridge_pct": s_p["overall_bridge_mean"],
        }
        for e in ALL_EVAL_ENVS:
            row[f"{e}_corral_pct"] = s_p[e]["corral_mean"]
            row[f"{e}_bridge_pct"] = s_p[e]["bridge_mean"]
        row["min_preset_corral_pct"] = min(row[f"{e}_corral_pct"] for e in SELECTABLE_ENV_IDS)
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(os.path.join(output_dir, "summary_30_runs.csv"), index=False)

    n_runs = len(df_sum)
    u_sub = untrained_list[:n_runs]

    def desc(arr: np.ndarray) -> dict:
        ci_lo, ci_hi = bootstrap_mean_ci(arr)
        sw_s, sw_p, sw_note = safe_shapiro(arr)
        return {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "shapiro_W": None if np.isnan(sw_s) else sw_s,
            "shapiro_p": sw_p,
            "shapiro_note": sw_note,
        }

    metrics_stats = {
        "overall_corral_pct": desc(df_sum["overall_corral_pct"].values),
        "overall_bridge_pct": desc(df_sum["overall_bridge_pct"].values),
        "final_best_fitness": desc(df_sum["final_best_fitness"].values),
        "val_fitness": desc(df_sum["val_fitness"].values),
        "train_minutes": desc(df_sum["train_minutes"].values),
    }
    for e in ALL_EVAL_ENVS:
        trained_c = df_sum[f"{e}_corral_pct"].values
        trained_b = df_sum[f"{e}_bridge_pct"].values
        untrained_c = np.array([u["eval"][primary_key][e]["corral_mean"] for u in u_sub], dtype=np.float64)
        metrics_stats[f"{e}_corral_pct"] = desc(trained_c)
        metrics_stats[f"{e}_bridge_pct"] = desc(trained_b)
        metrics_stats[f"{e}_untrained_corral_pct"] = desc(untrained_c)
        metrics_stats[f"{e}_vs_untrained"] = safe_mannwhitney(trained_c, untrained_c)

    # Seed reliability criteria
    n_ge_80_all3 = int((df_sum["min_preset_corral_pct"] >= 80.0).sum())
    n_ge_90_all3 = int((df_sum["min_preset_corral_pct"] >= 90.0).sum())

    # Friedman test across the 3 selectable scenarios (Classic, High->South, Low->North)
    c1 = df_sum["classic_corral_pct"].values
    c2 = df_sum["north_south_corral_pct"].values
    c3 = df_sum["south_north_corral_pct"].values
    if n_runs >= 2 and not (np.allclose(c1, c2) and np.allclose(c2, c3)):
        fried_stat, fried_p = scipy_stats.friedmanchisquare(c1, c2, c3)
        fried_stat, fried_p = float(fried_stat), float(fried_p)
    else:
        fried_stat, fried_p = 0.0, 1.0

    pairwise = {
        "classic_vs_north_south": safe_wilcoxon(c1, c2),
        "classic_vs_south_north": safe_wilcoxon(c1, c3),
        "north_south_vs_south_north": safe_wilcoxon(c2, c3),
    }
    for v in pairwise.values():
        v["p_bonferroni"] = min(1.0, v["p_value"] * 3.0)

    report = {
        "timestamp": datetime.now().isoformat(),
        "completed_runs": n_runs,
        "target_runs": args.num_runs,
        "eval_horizon_steps": args.steps,
        "eval_trials_per_env": args.eval_trials,
        "reliability": {
            "seeds_min_all3_ge_80pct": n_ge_80_all3,
            "seeds_min_all3_ge_90pct": n_ge_90_all3,
            "pct_ge_80": 100.0 * n_ge_80_all3 / max(1, n_runs),
            "pct_ge_90": 100.0 * n_ge_90_all3 / max(1, n_runs),
        },
        "friedman_across_3_scenarios": {"chi2": fried_stat, "p_value": fried_p},
        "pairwise_wilcoxon": pairwise,
        "metrics": metrics_stats,
    }
    with open(os.path.join(output_dir, "statistical_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # 300-DPI Figure 1: Convergence with Mean +/- 95% Bootstrapped CI / IQR
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=300)
    gens = sorted(df_all["generation"].unique())
    best_med, best_q25, best_q75, best_mean = [], [], [], []
    corral_med, corral_q25, corral_q75, bridge_mean = [], [], [], []

    for g in gens:
        sub = df_all[df_all["generation"] == g]
        bf = sub["best_fitness"].values
        best_mean.append(np.mean(bf))
        best_med.append(np.median(bf))
        best_q25.append(np.percentile(bf, 25))
        best_q75.append(np.percentile(bf, 75))
        cr = sub["best_corral_ratio"].values * 100.0
        corral_med.append(np.median(cr))
        corral_q25.append(np.percentile(cr, 25))
        corral_q75.append(np.percentile(cr, 75))
        bridge_mean.append(np.mean(sub["best_bridge_ratio"].values * 100.0))

    s1_end = df_all[df_all["stage"] == 1]["generation"].max()
    ax1.plot(gens, best_med, label="Median Best Fitness", color="#1f77b4", lw=2.2)
    ax1.plot(gens, best_mean, label="Mean Best Fitness", color="#ff7f0e", ls="--", lw=1.7)
    ax1.fill_between(gens, best_q25, best_q75, color="#1f77b4", alpha=0.22, label="IQR (25th–75th pct)")
    if pd.notna(s1_end):
        ax1.axvline(s1_end + 0.5, color="#444444", ls=":", lw=1.4, label=f"S1→S2 (Gen {int(s1_end)})")
    ax1.set_title(f"(a) {n_runs}-Run Convergence Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Balanced Fitness")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.8)

    ax2.plot(gens, bridge_mean, label="Mean Bridge Crossing (%)", color="#2ca02c", lw=2.0)
    ax2.plot(gens, corral_med, label="Median Goal Corral (%)", color="#d62728", lw=2.2)
    ax2.fill_between(gens, corral_q25, corral_q75, color="#d62728", alpha=0.22, label="Corral IQR")
    if pd.notna(s1_end):
        ax2.axvline(s1_end + 0.5, color="#444444", ls=":", lw=1.4)
    ax2.set_title(f"(b) {n_runs}-Run Task Completion Milestones", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(-2, 106)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_30_convergence.png"), dpi=300)
    plt.close()

    # 300-DPI Figure 2: Scenario Boxplots + Individual Seed Dots
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=300)
    data_list = [df_sum[f"{e}_corral_pct"].values for e in ALL_EVAL_ENVS] + [
        np.array([u["eval"][primary_key]["classic"]["corral_mean"] for u in u_sub], dtype=np.float64)
    ]
    bp = ax.boxplot(data_list, patch_artist=True, widths=0.52, showfliers=False)
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#9ca3af"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set(color="#111111", linewidth=2.0)

    rng_j = np.random.default_rng(7)
    for i, vals in enumerate(data_list, 1):
        jitter = rng_j.normal(0.0, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="#1f2937", s=18, alpha=0.75, zorder=3)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Scenario 1\n(Classic)", "Scenario 2\n(High→South)", "Scenario 3\n(Low→North)", "Random\nVariants", "Untrained\nBaseline"])
    ax.set_ylabel("Zero-Shot Goal Corral Rate (%)", fontsize=11)
    ax.set_ylim(-5, 108)
    ax.set_title(f"Zero-Shot Spatial Generalization Across {n_runs} Independent Seeds", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_30_scenario_boxplots.png"), dpi=300)
    plt.close()


def run_champion_scalability(best_ckpt_path: str, args, output_dir: str):
    # Runs final N=10,18,30,50 scalability benchmark (1500 & 2400 steps) on the overall 30-run champion.
    brain = HerdingTransformerBrain()
    brain.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
    genome = brain.get_flat_genome()

    env_cycle = list(SELECTABLE_ENV_IDS) + ["random"]
    jobs, meta = [], []
    for size in [10, 18, 30, 50]:
        for t in range(args.eval_trials):
            eval_seed = 9999 + size * 100 + t * 7
            env_spec = env_cycle[t % len(env_cycle)]
            jobs.append((genome, [("full", args.steps, eval_seed, env_spec, eval_seed + 5000)], size))
            meta.append((args.steps, size, env_spec))
            if size in (30, 50):
                ext_steps = max(args.steps, 2400)
                jobs.append((genome, [("full", ext_steps, eval_seed, env_spec, eval_seed + 5000)], size))
                meta.append((ext_steps, size, env_spec))

    with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
        results = pool.map(evaluate_genome_batch, jobs, chunksize=1)

    recs = []
    for (horizon, size, env_spec), res in zip(meta, results):
        recs.append(
            {
                "horizon_steps": horizon,
                "herd_size": size,
                "env_spec": env_spec,
                "corral_pct": res["corral_ratio"] * 100.0,
                "bridge_pct": res["bridge_ratio"] * 100.0,
                "flock_spread": res["flock_spread"],
            }
        )
    df_sc = pd.DataFrame(recs)
    df_sc.to_csv(os.path.join(output_dir, "champion_scalability.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Resumable 30-Run Statistical Experiment & Nonparametric Analysis Suite"
    )
    parser.add_argument("--num-runs", type=int, default=30, help="Number of independent random seeds (default: 30)")
    parser.add_argument("--start-seed", type=int, default=1001, help="First random seed (default: 1001)")
    parser.add_argument("--generations", type=int, default=90, help="Generations per run (default: 90)")
    parser.add_argument("--pop-size", type=int, default=GA_POPULATION_SIZE, help=f"ES population size (default: {GA_POPULATION_SIZE})")
    parser.add_argument("--steps", type=int, default=1500, help="Simulation steps per training/primary eval episode (default: 1500)")
    parser.add_argument("--eval-trials", type=int, default=8, help="Unseen test trials per scenario per seed (default: 8)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2), help="Parallel CPU workers")
    parser.add_argument("--sheep", type=int, default=SHEEP_COUNT, help=f"Training flock size (default: {SHEEP_COUNT})")
    parser.add_argument("--env", type=str, default="multi", choices=["multi", "classic", "north_south", "south_north"], help="Environment mode (default: multi)")
    parser.add_argument("--output-dir", type=str, default="statistical_30_runs", help="Directory for logs, checkpoints, stats, and figures")
    parser.add_argument("--update-best", action="store_true", help="Copy best overall champion across all runs to best_dog_genome.pt")
    parser.add_argument("--dry-run", action="store_true", help="Quick 2-run pipeline verification (3 gens, pop=6, 200 steps)")

    args = parser.parse_args()
    env_mode = resolve_env_id(args.env)

    if args.dry_run:
        print("[30-Run Suite] Dry-run enabled: 2 runs, 3 generations, pop=6, 200 steps, 2 eval trials.")
        args.num_runs = 2
        args.generations = 3
        args.pop_size = 6
        args.steps = 200
        args.eval_trials = 2

    os.makedirs(args.output_dir, exist_ok=True)
    runs_root = os.path.join(args.output_dir, "runs")
    os.makedirs(runs_root, exist_ok=True)

    print("=" * 78)
    print(" 30-RUN STATISTICAL EVALUATION HARNESS (Curriculum OpenAI-ES) ")
    print("=" * 78)
    print(f"Start Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs / Seeds:      {args.num_runs} independent runs ({args.start_seed}..{args.start_seed + args.num_runs - 1})")
    print(f"Schedule / Budget: {args.generations} gens x pop {args.pop_size} x {args.steps} steps | {args.workers} CPU workers")
    print(f"Env Mode:          {env_mode} | Eval Suite: 4 map types x {args.eval_trials} paired trials")
    print(f"Output Directory:  '{args.output_dir}/'")
    print("=" * 78)

    untrained_list = get_or_compute_untrained_baseline(args, args.output_dir)

    all_dfs: list[pd.DataFrame] = []
    all_evals: list[dict] = []
    suite_start = time.time()
    fresh_times: list[float] = []

    best_score = -1e9
    best_model_src = None

    for idx in range(1, args.num_runs + 1):
        seed = args.start_seed + idx - 1
        run_dir = os.path.join(runs_root, f"run_{idx:02d}_seed_{seed}")
        was_cached = os.path.exists(os.path.join(run_dir, "DONE"))

        df_seed, ev = run_single_experiment(idx, seed, args, env_mode, run_dir)
        all_dfs.append(df_seed)
        all_evals.append(ev)

        if not was_cached:
            fresh_times.append(ev["train_minutes"])

        if ev["val_fitness"] > best_score:
            best_score = ev["val_fitness"]
            best_model_src = os.path.join(run_dir, "best_model.pt")

        # Update cumulative statistics & plots after every run
        generate_statistical_outputs(all_dfs, all_evals, untrained_list, args, args.output_dir)

        rem = args.num_runs - idx
        if fresh_times and rem > 0:
            eta_min = np.mean(fresh_times) * rem
            print(f"[Progress] {idx}/{args.num_runs} complete | Est. remaining: {eta_min/60:.2f} hours ({eta_min:.0f} min)")

    best_dst = os.path.join(args.output_dir, "best_30run_champion.pt")
    if best_model_src and os.path.exists(best_model_src):
        state = torch.load(best_model_src, map_location="cpu")
        torch.save(state, best_dst)
        if args.update_best and not args.dry_run:
            torch.save(state, "best_dog_genome.pt")
            print("[Champion] Updated root 'best_dog_genome.pt' with 30-run global champion.")
        run_champion_scalability(best_dst, args, args.output_dir)

    total_hours = (time.time() - suite_start) / 3600.0
    print("\n" + "=" * 78)
    print(f" SUITE COMPLETE ({len(all_evals)}/{args.num_runs} runs) in {total_hours:.2f} hours ")
    print(f" Statistical JSON:   {args.output_dir}/statistical_report.json")
    print(f" Summary CSV:        {args.output_dir}/summary_30_runs.csv")
    print(f" Global Champion:    {best_dst} (Val Fitness: {best_score:.1f})")
    print("=" * 78)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
