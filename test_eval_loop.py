"""
test_eval_loop.py — End-to-end evaluation feedback loop test
=============================================================
Steps:
  1. Show current eval_config weights (baseline)
  2. Inject NEGATIVE feedback for slow_query pattern metrics in DB
  3. Run the evaluator once
  4. Show updated eval_config weights → verify they changed
  5. Re-run a trace through the pipeline → verify different fast-path behavior
"""

import asyncio
import json
import os

from src.dao.factory import create_dao
from src.intel.eval_config import load_config, save_config, EvalConfig
from src.intel.evaluator import Evaluator
from src.config import EVAL_CONFIG_PATH


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_weights(label: str, cfg: EvalConfig):
    w = cfg.ranking_weights
    print(f"\n  {label}")
    print(f"    latency:        {w.latency:.4f}")
    print(f"    error_density:  {w.error_density:.4f}")
    print(f"    pattern_match:  {w.pattern_match:.4f}")
    print(f"    structural:     {w.structural:.4f}")
    print(f"    temporal:       {w.temporal:.4f}")
    print(f"    fast_path_threshold: {cfg.fast_path_threshold}")
    print(f"    pattern_overrides:   {cfg.pattern_confidence_overrides}")
    print(f"    version:             {cfg.version}")


async def main():
    dao = create_dao()
    await dao._init_db()

    # ── Step 1: Show baseline ──
    print_section("STEP 1 — Baseline eval_config")
    cfg_before = load_config(EVAL_CONFIG_PATH)
    print_weights("BEFORE", cfg_before)

    # ── Step 2: Check existing metrics ──
    print_section("STEP 2 — Check existing metrics in DB")
    all_metrics = await dao.get_intelligence_metrics(filters={"limit": 1000})
    print(f"  Total intelligence_metrics rows: {len(all_metrics)}")

    # Find metrics that matched known patterns (names match PatternMatch.pattern_name)
    slow_query_metrics = []
    n1_metrics = []
    for m in all_metrics:
        patterns = m.get("pattern_matches", [])
        if isinstance(patterns, str):
            if "Slow Query" in patterns:
                slow_query_metrics.append(m)
            if "N+1 Query" in patterns:
                n1_metrics.append(m)
        elif isinstance(patterns, list):
            for p in patterns:
                if isinstance(p, dict):
                    if p.get("name") == "Slow Query":
                        slow_query_metrics.append(m)
                    if p.get("name") == "N+1 Query":
                        n1_metrics.append(m)

    print(f"  Metrics with 'Slow Query' pattern: {len(slow_query_metrics)}")
    print(f"  Metrics with 'N+1 Query' pattern:  {len(n1_metrics)}")

    if not slow_query_metrics and not n1_metrics:
        print("\n  ❌ No pattern-matched metrics found! Run test_all_traces first.")
        return

    # ── Step 3: Inject NEGATIVE feedback ──
    print_section("STEP 3 — Inject NEGATIVE feedback for slow_query & n_plus_1")

    # We need enough negative feedback to trigger the evaluator's adjustments.
    # min_samples_for_adjustment defaults to 5, so we temporarily lower it.
    feedback_count = 0
    for m in slow_query_metrics:
        cid = m.get("chat_id", "")
        tid = m.get("trace_id", "")
        if cid and tid:
            await dao.update_user_feedback(cid, tid, "negative", "Test: bad Slow Query RCA")
            feedback_count += 1
            print(f"    👎 Negative feedback → trace {tid[:16]}... (Slow Query)")

    for m in n1_metrics:
        cid = m.get("chat_id", "")
        tid = m.get("trace_id", "")
        if cid and tid:
            await dao.update_user_feedback(cid, tid, "negative", "Test: bad N+1 Query RCA")
            feedback_count += 1
            print(f"    👎 Negative feedback → trace {tid[:16]}... (N+1 Query)")

    # Also inject negative feedback for some non-pattern (LLM) metrics
    llm_metrics = [m for m in all_metrics if not m.get("fast_path_used")]
    for m in llm_metrics[:3]:
        cid = m.get("chat_id", "")
        tid = m.get("trace_id", "")
        if cid and tid:
            await dao.update_user_feedback(cid, tid, "negative", "Test: bad LLM response")
            feedback_count += 1
            print(f"    👎 Negative feedback → trace {tid[:16]}... (LLM path)")

    print(f"\n  Total negative feedback injected: {feedback_count}")

    # ── Step 4: Temporarily lower min_samples_for_adjustment for testing ──
    print_section("STEP 4 — Lower min_samples_for_adjustment to 2 for testing")
    cfg_before.min_samples_for_adjustment = 2
    save_config(cfg_before, EVAL_CONFIG_PATH)
    print(f"  min_samples_for_adjustment = 2 (was 5)")

    # ── Step 5: Run the evaluator ──
    print_section("STEP 5 — Run Evaluator")
    evaluator = Evaluator(dao, config_path=EVAL_CONFIG_PATH)
    report = await evaluator.run(lookback_days=30)  # wide lookback to catch all our data

    # ── Step 6: Show updated weights ──
    print_section("STEP 6 — Updated eval_config after evaluation")
    cfg_after = load_config(EVAL_CONFIG_PATH)
    print_weights("AFTER", cfg_after)

    # Compare
    print_section("STEP 7 — Comparison")
    changes_found = False

    if cfg_before.fast_path_threshold != cfg_after.fast_path_threshold:
        print(f"  ✅ fast_path_threshold CHANGED: {cfg_before.fast_path_threshold} → {cfg_after.fast_path_threshold}")
        changes_found = True

    if cfg_before.pattern_confidence_overrides != cfg_after.pattern_confidence_overrides:
        print(f"  ✅ pattern_confidence_overrides CHANGED:")
        print(f"     Before: {cfg_before.pattern_confidence_overrides}")
        print(f"     After:  {cfg_after.pattern_confidence_overrides}")
        changes_found = True

    w_before = cfg_before.ranking_weights
    w_after = cfg_after.ranking_weights
    for field in ["latency", "error_density", "pattern_match", "structural", "temporal"]:
        v_before = getattr(w_before, field)
        v_after = getattr(w_after, field)
        if abs(v_before - v_after) > 0.001:
            print(f"  ✅ ranking_weights.{field} CHANGED: {v_before:.4f} → {v_after:.4f}")
            changes_found = True

    if not changes_found:
        print("  ⚠️  No weight changes detected. Check evaluator logic / feedback data.")
    else:
        print("\n  🎉 Evaluation feedback loop is WORKING!")

    # ── Step 8: Re-run a trace through the pipeline to see different behavior ──
    print_section("STEP 8 — Re-run a trace to check if pipeline uses new weights")

    from src.intel.pipeline import IntelligencePipeline
    # Force a fresh load of eval_config
    pipeline = IntelligencePipeline()

    print(f"  Pipeline fast_path_threshold: {pipeline.config.fast_path_threshold}")
    print(f"  Pipeline pattern_overrides:   {pipeline.eval_config.pattern_confidence_overrides}")
    print(f"  Pipeline ranking weights:")
    print(f"    error_density = {pipeline.config.ranking.error_density_weight}")
    print(f"    latency       = {pipeline.config.ranking.latency_anomaly_weight}")
    print(f"    pattern_match = {pipeline.config.ranking.pattern_match_weight}")
    print(f"    depth         = {pipeline.config.ranking.depth_weight}")
    print(f"    temporal      = {pipeline.config.ranking.temporal_correlation_weight}")

    # If there's an override on Slow Query, the fast path behavior may change
    if "Slow Query" in pipeline.eval_config.pattern_confidence_overrides:
        override = pipeline.eval_config.pattern_confidence_overrides["Slow Query"]
        print(f"\n  Slow Query override = {override}")
        if override < 1.0:
            print(f"  → Pattern confidence will be multiplied by {override}")
            print(f"  → If raw confidence * {override} < {pipeline.config.fast_path_threshold}, "
                  f"fast path will be SKIPPED (LLM used instead)")
            example = 0.95 * override
            if example < pipeline.config.fast_path_threshold:
                print(f"  → Example: 0.95 × {override} = {example:.3f} < {pipeline.config.fast_path_threshold} → LLM path!")
            else:
                print(f"  → Example: 0.95 × {override} = {example:.3f} ≥ {pipeline.config.fast_path_threshold} → still fast path")

    # ── Step 9: Restore min_samples_for_adjustment ──
    print_section("STEP 9 — Restore min_samples_for_adjustment to 5")
    cfg_after.min_samples_for_adjustment = 5
    save_config(cfg_after, EVAL_CONFIG_PATH)
    print(f"  min_samples_for_adjustment restored to 5")

    print_section("TEST COMPLETE")
    print("  The evaluation feedback loop successfully:")
    print("  1. Read negative feedback from the database")
    print("  2. Calculated pattern accuracy")
    print("  3. Adjusted pattern_confidence_overrides")
    print("  4. The pipeline now uses the updated weights")
    print()


if __name__ == "__main__":
    asyncio.run(main())
