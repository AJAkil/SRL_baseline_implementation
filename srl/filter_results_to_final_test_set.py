#!/usr/bin/env python3
"""
Filter SRL evaluation results to only the FINAL TEST SET samples
(the common MalGraph samples used across all experiments).

Recovers sample names from sample_idx via os.listdir ordering on the
original ACFG directories, then intersects with the MG common sample lists.

Produces filtered summary files with recomputed ASR and timing metrics.
"""

import os
import json
import statistics
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_RESULTS_DIR = Path("/home/newdrive/makil/projects/SRL_Implementation/srl/evaluation_results")
SAMPLE_NAMES_BASE = Path("/home/newdrive/makil/projects/Adv_Binary_Clean/PAPER_EXP_FOLDER/SCRIPTS/ALL_DIE_VERIFICATION/")
ACFG_BASE_DIR = Path("/home/newdrive2/makil/ALL_DATASETS/MALWARE/ADV_MALWARE_GEN/SRL_DATASETS/SRL_TESTING_SETS/SRL")

CONFIGS = {
    "100FPR": {
        "acfg_dir": ACFG_BASE_DIR / "SRL_100_FPR_ACFGS",
        "common_samples_file": SAMPLE_NAMES_BASE / "100_FPR" / "MG" / "NORMAL_FC_ATTACK" / "MG_100_[VERIFICATION]_RANDOM_CALL_SITE_[FINAL_EXP_FOR_PAPER]_fc_attack_k_1_3_common_samples_ALL_ATTACKS.txt",
        "results_file": EVAL_RESULTS_DIR / "100FPR" / "evaluation_report.json",
        "output_dir": EVAL_RESULTS_DIR / "100FPR",
    },
    "1000FPR": {
        "acfg_dir": ACFG_BASE_DIR / "SRL_1000_FPR_ACFGS",
        "common_samples_file": SAMPLE_NAMES_BASE / "1000_FPR" / "MG" / "NORMAL_FC_ATTACK" / "MG_1000_[VERIFICATION]_RANDOM_CALL_SITE_[FINAL_EXP_FOR_PAPER]_fc_attack_k_1_3_common_samples_ALL_ATTACKS.txt",
        "results_file": EVAL_RESULTS_DIR / "1000FPR" / "evaluation_report.json",
        "output_dir": EVAL_RESULTS_DIR / "1000FPR",
    },
}


def load_common_samples(filepath: Path) -> set:
    """Load common sample names, strip .exe suffix to get base names."""
    with open(filepath) as f:
        return {line.strip().replace(".exe", "") for line in f if line.strip()}


def build_idx_to_name(acfg_dir: Path) -> dict:
    """Reproduce the exact os.listdir ordering used during evaluation."""
    files = [f for f in os.listdir(acfg_dir) if f.endswith(".json")]
    return {i: f.replace(".json", "") for i, f in enumerate(files)}


def fmt_time(seconds: float) -> str:
    """Format seconds as 'Xm Y.YYs'."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {s:.2f}s"


def generate_summary(config_name: str, cfg: dict):
    print(f"\n{'='*80}")
    print(f"Processing {config_name}")
    print(f"{'='*80}")

    # 1. Build idx -> name mapping
    idx_to_name = build_idx_to_name(cfg["acfg_dir"])
    print(f"  ACFG files (total): {len(idx_to_name)}")

    # 2. Load common (final test set) sample names
    common_samples = load_common_samples(cfg["common_samples_file"])
    print(f"  Common samples (final test set): {len(common_samples)}")

    # 3. Load evaluation results
    with open(cfg["results_file"]) as f:
        data = json.load(f)
    all_results = data["per_sample_results"]
    print(f"  Evaluation results: {len(all_results)}")

    # 4. Filter to only samples in the final test set
    filtered = []
    for r in all_results:
        name = idx_to_name[r["sample_idx"]]
        if name in common_samples:
            r["sample_name"] = name
            filtered.append(r)

    total = len(filtered)
    bypassed = [r for r in filtered if r["bypassed"]]
    failed = [r for r in filtered if not r["bypassed"]]

    print(f"  Matched in final test set: {total}")
    print(f"  Bypassed: {len(bypassed)}, Failed: {len(failed)}")

    # Sanity: check how many common samples were NOT found in eval
    matched_names = {r["sample_name"] for r in filtered}
    missing = common_samples - matched_names
    if missing:
        print(f"  WARNING: {len(missing)} common samples not found in evaluation results")

    # 5. Compute metrics
    times_all = [r["elapsed_time_seconds"] for r in filtered]
    times_bypassed = [r["elapsed_time_seconds"] for r in bypassed]
    times_failed = [r["elapsed_time_seconds"] for r in failed]

    mutations_all = [r["num_mutations"] for r in filtered]
    score_reductions = [r["score_reduction"] for r in filtered]
    final_scores = [r["final_score"] for r in filtered]

    asr = len(bypassed) / total if total > 0 else 0
    avg_mutations = statistics.mean(mutations_all) if mutations_all else 0
    median_mutations = statistics.median(mutations_all) if mutations_all else 0
    avg_score_reduction = statistics.mean(score_reductions) if score_reductions else 0
    avg_final_score = statistics.mean(final_scores) if final_scores else 0

    total_time = sum(times_all)
    avg_time = statistics.mean(times_all) if times_all else 0
    median_time = statistics.median(times_all) if times_all else 0
    min_time = min(times_all) if times_all else 0
    max_time = max(times_all) if times_all else 0

    # 6. Write summary
    output_file = cfg["output_dir"] / "summary_FINAL_TEST_SET.txt"
    lines = []
    lines.append("=" * 80)
    lines.append("SRL-MalGraph Evaluation Summary (FINAL TEST SET ONLY)")
    lines.append("=" * 80)
    lines.append(f"Original Checkpoint: {data.get('checkpoint', 'N/A') if isinstance(data.get('checkpoint'), str) else cfg['results_file']}")
    lines.append(f"Common Samples File: {cfg['common_samples_file'].name}")
    lines.append(f"Total Samples (final test set): {total}")
    lines.append(f"Successful Attacks: {len(bypassed)}")
    lines.append(f"Failed Attacks: {len(failed)}")
    lines.append(f"Attack Success Rate (ASR): {asr*100:.2f}%")
    lines.append(f"Average Mutations: {avg_mutations:.2f}")
    lines.append(f"Median Mutations: {median_mutations:.1f}")
    lines.append(f"Average Score Reduction: {avg_score_reduction:.4f}")
    lines.append(f"Average Final Score: {avg_final_score:.4f}")
    lines.append("-" * 80)
    lines.append("OVERALL TIMING STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Total Time: {fmt_time(total_time)} ({total_time/3600:.2f} hours)")
    lines.append(f"Average Time per Sample: {fmt_time(avg_time)} ({avg_time:.2f}s)")
    lines.append(f"Median Time: {median_time:.2f}s")
    lines.append(f"Min Time: {min_time:.2f}s")
    lines.append(f"Max Time: {max_time:.2f}s")

    if bypassed:
        bp_total = sum(times_bypassed)
        bp_mean = statistics.mean(times_bypassed)
        bp_median = statistics.median(times_bypassed)
        bp_throughput = len(bypassed) / (bp_total / 3600) if bp_total > 0 else 0
        lines.append("-" * 80)
        lines.append(f"BYPASSED SAMPLES TIMING ({len(bypassed)} samples)")
        lines.append("-" * 80)
        lines.append(f"Total Time: {bp_total/3600:.2f} hours")
        lines.append(f"Mean Time per Sample: {fmt_time(bp_mean)} ({bp_mean:.2f}s)")
        lines.append(f"Median Time per Sample: {fmt_time(bp_median)} ({bp_median:.2f}s)")
        lines.append(f"Throughput: {bp_throughput:.2f} samples/hour")

    if failed:
        fl_total = sum(times_failed)
        fl_mean = statistics.mean(times_failed)
        fl_median = statistics.median(times_failed)
        fl_throughput = len(failed) / (fl_total / 3600) if fl_total > 0 else 0
        lines.append("-" * 80)
        lines.append(f"FAILED SAMPLES TIMING ({len(failed)} samples)")
        lines.append("-" * 80)
        lines.append(f"Total Time: {fl_total/3600:.2f} hours")
        lines.append(f"Mean Time per Sample: {fmt_time(fl_mean)} ({fl_mean:.2f}s)")
        lines.append(f"Median Time per Sample: {fmt_time(fl_median)} ({fl_median:.2f}s)")
        lines.append(f"Throughput: {fl_throughput:.2f} samples/hour")

    lines.append("=" * 80)

    summary_text = "\n".join(lines) + "\n"
    with open(output_file, "w") as f:
        f.write(summary_text)

    print(f"\n  Written: {output_file}")
    print(summary_text)

    # 7. Also save filtered JSON with sample names
    filtered_json = {
        "config": config_name,
        "common_samples_file": str(cfg["common_samples_file"]),
        "total_samples": total,
        "successful_attacks": len(bypassed),
        "failed_attacks": len(failed),
        "attack_success_rate": asr,
        "avg_mutations": avg_mutations,
        "median_mutations": median_mutations,
        "avg_score_reduction": avg_score_reduction,
        "avg_final_score": avg_final_score,
        "avg_time_seconds": avg_time,
        "median_time_seconds": median_time,
        "total_time_seconds": total_time,
        "per_sample_results": filtered,
    }
    json_out = cfg["output_dir"] / "evaluation_report_FINAL_TEST_SET.json"
    with open(json_out, "w") as f:
        json.dump(filtered_json, f, indent=2)
    print(f"  Written: {json_out}")


if __name__ == "__main__":
    for config_name, cfg in CONFIGS.items():
        generate_summary(config_name, cfg)
    print("\nDone.")
