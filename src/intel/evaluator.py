"""
Evaluation Feedback Loop — Stage 9
=====================================
Scheduled job (default: weekly) that reads user feedback from the
database, calculates pattern accuracy, adjusts ranking weights, and
writes an updated config for the next pipeline run.

WHAT IT DOES
  1. Reads all intelligence_metrics with user_feedback from the past N days
  2. Calculates accuracy per pattern  (positive / total rated)
  3. Flags: HEALTHY (≥80%) | DEGRADED (60-80%) | CRITICAL (<60%) | DISABLED (<20%)
  4. Adjusts pattern_confidence_overrides in eval_config.json
  5. Checks fast-path accuracy vs full-path accuracy
  6. Writes updated eval_config.json + weekly report JSON

WHAT IT DOES NOT DO
  - No LLM calls
  - No prompt changes
  - No model retraining
  It only adjusts WEIGHTS and CONFIDENCE MULTIPLIERS

HOW TO RUN
  One-shot (manual or cron):
    python -m src.intel.evaluator

  Scheduled loop (production — uses EVAL_SCHEDULE_SECONDS from config):
    python -m src.intel.evaluator --loop

  Custom interval (override):
    python -m src.intel.evaluator --loop 3600        # every hour
    python -m src.intel.evaluator --lookback 14       # 14-day window

  From code:
    evaluator = Evaluator(dao)
    report = await evaluator.run()
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any

from src.intel.eval_config import (
    EvalConfig,
    load_config,
    save_config,
)
from src.dao.factory import create_dao

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "evaluation_reports")


class PatternReport:
    """Accuracy report for a single pattern."""

    def __init__(self, name: str, total: int, positive: int, negative: int, unrated: int):
        self.name = name
        self.total = total
        self.positive = positive
        self.negative = negative
        self.unrated = unrated
        self.rated = positive + negative
        self.accuracy = positive / max(self.rated, 1)
        self.status = ""      # Set by evaluator
        self.action = ""      # Set by evaluator
        self.old_override = 1.0
        self.new_override = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total": self.total,
            "positive": self.positive,
            "negative": self.negative,
            "unrated": self.unrated,
            "accuracy": round(self.accuracy, 4),
            "status": self.status,
            "action": self.action,
            "old_override": round(self.old_override, 4),
            "new_override": round(self.new_override, 4),
        }


class EvaluationReport:
    """Full weekly evaluation report."""

    def __init__(self):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.week = datetime.now(timezone.utc).strftime("%Y-W%W")
        self.total_queries = 0
        self.total_with_feedback = 0
        self.feedback_rate = 0.0
        self.pattern_reports: list[PatternReport] = []
        self.fast_path_accuracy = 0.0
        self.full_path_accuracy = 0.0
        self.fast_path_adjustment = None
        self.config_changes: list[str] = []
        self.recommendations: list[str] = []

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "week": self.week,
            "total_queries": self.total_queries,
            "total_with_feedback": self.total_with_feedback,
            "feedback_rate": round(self.feedback_rate, 4),
            "pattern_health": {p.name: p.to_dict() for p in self.pattern_reports},
            "fast_path_accuracy": round(self.fast_path_accuracy, 4),
            "full_path_accuracy": round(self.full_path_accuracy, 4),
            "fast_path_adjustment": self.fast_path_adjustment,
            "config_changes": self.config_changes,
            "recommendations": self.recommendations,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """Weekly evaluation loop that adjusts the pipeline config based on feedback.

    Usage:
        dao = create_dao()
        evaluator = Evaluator(dao)
        report = await evaluator.run()
    """

    # All known pattern types to evaluate.
    # These MUST match the pattern_name values written by the pattern
    # classes (e.g. SlowQueryPattern.pattern_name = "Slow Query") and
    # serialized into the intelligence_metrics.pattern_matches JSON.
    KNOWN_PATTERNS = [
        "Slow Query",
        "N+1 Query",
        "Retry Storm",
        "Timeout",
        "Connection Pool",
        "Rate Limit",
        "Deadlock",
        "Memory Leak",
        "Connection Refused",
    ]

    def __init__(self, dao, config_path: str | None = None):
        """
        Args:
            dao: Database client (TraceRootSQLiteClient or TraceRootMongoDBClient)
            config_path: Path to eval_config.json (None = default location)
        """
        self.dao = dao
        self.config_path = config_path
        self.config = load_config(config_path) if config_path else load_config()

    async def run(self, lookback_days: int = 7) -> EvaluationReport:
        """Run the full evaluation loop.

        Args:
            lookback_days: How many days of feedback to analyze (default: 7)

        Returns:
            EvaluationReport with all findings and changes made
        """
        logger.info("="*60)
        logger.info(
            "EVALUATION LOOP — %s  |  Lookback: %d days",
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            lookback_days,
        )
        logger.info("="*60)

        report = EvaluationReport()

        # ── Step 1: Fetch all metrics from the lookback period ──
        all_metrics = await self._fetch_metrics(lookback_days)
        report.total_queries = len(all_metrics)

        rated_metrics = [m for m in all_metrics if m.get("user_feedback") is not None]
        report.total_with_feedback = len(rated_metrics)
        report.feedback_rate = len(rated_metrics) / max(len(all_metrics), 1)

        logger.info(
            "Metrics — total: %d  |  with feedback: %d  |  rate: %.0f%%",
            report.total_queries, report.total_with_feedback,
            report.feedback_rate * 100,
        )

        if report.total_with_feedback < 3:
            logger.warning("Not enough feedback to evaluate (need ≥3, got %d). Skipping.",
                           report.total_with_feedback)
            report.recommendations.append(
                "Not enough feedback data. Encourage users to rate responses."
            )
            return report

        # ── Step 2: Analyze accuracy per pattern ──
        pattern_reports = self._analyze_patterns(all_metrics)
        report.pattern_reports = pattern_reports

        for pr in pattern_reports:
            logger.info(
                "Pattern %-20s  accuracy=%.0f%%  rated=%d  status=%s  action=%s",
                pr.name, pr.accuracy * 100, pr.rated, pr.status, pr.action,
            )

        # ── Step 3: Adjust pattern confidence overrides ──
        changes = self._adjust_pattern_overrides(pattern_reports)
        report.config_changes.extend(changes)

        # ── Step 4: Analyze fast path vs full path ──
        fp_acc, full_acc, fp_change = self._analyze_fast_path(rated_metrics)
        report.fast_path_accuracy = fp_acc
        report.full_path_accuracy = full_acc
        report.fast_path_adjustment = fp_change
        if fp_change:
            report.config_changes.append(fp_change)

        logger.info(
            "Fast path accuracy=%.0f%%  |  Full path accuracy=%.0f%%  |  adjustment=%s",
            fp_acc * 100, full_acc * 100, fp_change or "none",
        )

        # ── Step 5: Build recommendations ──
        recommendations = self._build_recommendations(pattern_reports, fp_acc, full_acc)
        report.recommendations = recommendations

        for rec in recommendations:
            logger.info("Recommendation: %s", rec)

        # ── Step 6: Save updated config ──
        self.config.last_evaluation = datetime.now(timezone.utc).isoformat()
        self.config.version += 1

        if self.config_path:
            save_config(self.config, self.config_path)
        else:
            save_config(self.config)

        # ── Step 7: Save report ──
        self._save_report(report)

        logger.info(
            "Config v%d saved. %d change(s) made.",
            self.config.version, len(report.config_changes),
        )

        return report

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: FETCH METRICS
    # ═══════════════════════════════════════════════════════════════════════════

    async def _fetch_metrics(self, lookback_days: int) -> list[dict]:
        """Fetch all intelligence_metrics from the lookback period."""
        all_metrics = await self.dao.get_intelligence_metrics(
            filters={"limit": 10000}
        )

        # Filter by date
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        recent = [
            m for m in all_metrics
            if m.get("timestamp", "") >= cutoff
        ]

        return recent

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: ANALYZE PATTERN ACCURACY
    # ═══════════════════════════════════════════════════════════════════════════

    def _analyze_patterns(self, all_metrics: list[dict]) -> list[PatternReport]:
        """Calculate accuracy for each known pattern."""
        reports = []

        for pattern_name in self.KNOWN_PATTERNS:
            # Count occurrences of this pattern
            total = 0
            positive = 0
            negative = 0
            unrated = 0

            for m in all_metrics:
                # Check if this pattern was matched in this query
                patterns = m.get("pattern_matches", [])
                if isinstance(patterns, str):
                    # SQLite stores as JSON string — check by substring
                    matched = pattern_name in patterns
                elif isinstance(patterns, list):
                    matched = any(
                        p.get("name", "") == pattern_name
                        for p in patterns
                        if isinstance(p, dict)
                    )
                else:
                    matched = False

                if not matched:
                    continue

                total += 1
                feedback = m.get("user_feedback")
                if feedback == "positive":
                    positive += 1
                elif feedback == "negative":
                    negative += 1
                else:
                    unrated += 1

            if total == 0:
                continue

            pr = PatternReport(
                name=pattern_name,
                total=total,
                positive=positive,
                negative=negative,
                unrated=unrated,
            )

            # Determine status
            thresholds = self.config.accuracy_thresholds
            rated = positive + negative

            if rated < self.config.min_samples_for_adjustment:
                pr.status = "insufficient_data"
                pr.action = f"Need {self.config.min_samples_for_adjustment - rated} more feedback"
            elif pr.accuracy >= thresholds.get("healthy", 0.80):
                pr.status = "healthy"
                pr.action = "No changes"
            elif pr.accuracy >= thresholds.get("degraded", 0.60):
                pr.status = "degraded"
                pr.action = "Lower confidence multiplier"
            elif pr.accuracy >= thresholds.get("disabled", 0.20):
                pr.status = "critical"
                pr.action = "Severely penalize"
            else:
                pr.status = "disabled"
                pr.action = "Disable pattern"

            reports.append(pr)

        return reports

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: ADJUST PATTERN OVERRIDES
    # ═══════════════════════════════════════════════════════════════════════════

    def _adjust_pattern_overrides(self, pattern_reports: list[PatternReport]) -> list[str]:
        """Adjust pattern_confidence_overrides based on accuracy.

        Returns list of change descriptions.
        """
        changes = []

        for pr in pattern_reports:
            old_override = self.config.pattern_confidence_overrides.get(pr.name, 1.0)
            pr.old_override = old_override

            if pr.status == "insufficient_data":
                pr.new_override = old_override  # Don't change
                continue

            elif pr.status == "healthy":
                # Good accuracy — restore trust if it was previously lowered
                if old_override < 1.0:
                    # Gradually restore: move 30% toward 1.0
                    new_override = old_override + (1.0 - old_override) * 0.3
                    new_override = min(new_override, 1.0)
                else:
                    new_override = 1.0

            elif pr.status == "degraded":
                # Moderate issues — lower to accuracy level
                # But don't lower more than 20% per evaluation
                target = max(pr.accuracy, 0.5)
                new_override = max(old_override - 0.2, target)

            elif pr.status == "critical":
                # Severe issues — drop to 0.3
                new_override = 0.3

            elif pr.status == "disabled":
                # Pattern is harmful — effectively disable
                new_override = 0.0

            else:
                new_override = old_override

            pr.new_override = round(new_override, 4)

            # Apply if changed
            if abs(new_override - old_override) > 0.01:
                self.config.pattern_confidence_overrides[pr.name] = round(new_override, 4)
                change = (
                    f"Pattern '{pr.name}': override {old_override:.2f} → {new_override:.2f} "
                    f"(accuracy: {pr.accuracy:.0%}, status: {pr.status})"
                )
                changes.append(change)

            # Remove override if back to 1.0 (clean up)
            if abs(new_override - 1.0) < 0.01 and pr.name in self.config.pattern_confidence_overrides:
                del self.config.pattern_confidence_overrides[pr.name]

        return changes

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: FAST PATH ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    def _analyze_fast_path(
        self,
        rated_metrics: list[dict],
    ) -> tuple[float, float, str | None]:
        """Compare fast path accuracy vs full path accuracy.

        If fast path is significantly worse, raise the threshold.

        Returns: (fast_path_accuracy, full_path_accuracy, adjustment_description)
        """
        fp_positive = 0
        fp_total = 0
        full_positive = 0
        full_total = 0

        for m in rated_metrics:
            is_fast_path = m.get("fast_path_used", False)
            is_positive = m.get("user_feedback") == "positive"

            if is_fast_path:
                fp_total += 1
                if is_positive:
                    fp_positive += 1
            else:
                full_total += 1
                if is_positive:
                    full_positive += 1

        fp_acc = fp_positive / max(fp_total, 1)
        full_acc = full_positive / max(full_total, 1)

        adjustment = None

        # Only adjust if we have enough samples
        if fp_total >= 5 and full_total >= 5:
            gap = full_acc - fp_acc

            if gap > 0.15:
                # Fast path is >15% worse than full path — raise threshold
                old_threshold = self.config.fast_path_threshold
                new_threshold = min(old_threshold + 0.05, 0.98)
                self.config.fast_path_threshold = round(new_threshold, 2)
                adjustment = (
                    f"Fast path threshold: {old_threshold:.2f} → {new_threshold:.2f} "
                    f"(fast path {fp_acc:.0%} vs full path {full_acc:.0%})"
                )

            elif gap < -0.05 and self.config.fast_path_threshold > 0.80:
                # Fast path is actually BETTER — lower threshold slightly
                old_threshold = self.config.fast_path_threshold
                new_threshold = max(old_threshold - 0.03, 0.80)
                self.config.fast_path_threshold = round(new_threshold, 2)
                adjustment = (
                    f"Fast path threshold: {old_threshold:.2f} → {new_threshold:.2f} "
                    f"(fast path is performing well: {fp_acc:.0%})"
                )

        return fp_acc, full_acc, adjustment

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_recommendations(
        self,
        pattern_reports: list[PatternReport],
        fp_accuracy: float,
        full_accuracy: float,
    ) -> list[str]:
        """Build human-readable recommendations."""
        recs = []

        # Critical patterns
        for pr in pattern_reports:
            if pr.status == "critical":
                recs.append(
                    f"CRITICAL: '{pr.name}' pattern has {pr.accuracy:.0%} accuracy "
                    f"({pr.negative} negative out of {pr.rated} rated). "
                    f"Review the detection logic in matcher.py."
                )
            elif pr.status == "disabled":
                recs.append(
                    f"DISABLED: '{pr.name}' pattern has been disabled "
                    f"({pr.accuracy:.0%} accuracy). Consider removing or rewriting it."
                )
            elif pr.status == "degraded":
                recs.append(
                    f"DEGRADED: '{pr.name}' pattern has {pr.accuracy:.0%} accuracy. "
                    f"Confidence multiplier lowered to {pr.new_override:.2f}."
                )

        # Previously penalized patterns that recovered
        for pr in pattern_reports:
            if pr.status == "healthy" and pr.old_override < 0.9:
                recs.append(
                    f"RECOVERED: '{pr.name}' pattern improved to {pr.accuracy:.0%}. "
                    f"Restoring confidence from {pr.old_override:.2f} to {pr.new_override:.2f}."
                )

        # Low feedback rate
        total_queries = sum(pr.total for pr in pattern_reports) if pattern_reports else 0
        total_rated = sum(pr.rated for pr in pattern_reports) if pattern_reports else 0
        if total_queries > 10 and total_rated / max(total_queries, 1) < 0.3:
            recs.append(
                "LOW FEEDBACK RATE: Only {:.0%} of queries received feedback. "
                "Consider adding a more prominent feedback UI.".format(
                    total_rated / max(total_queries, 1)
                )
            )

        # Overall accuracy
        total_positive = sum(pr.positive for pr in pattern_reports)
        total_negative = sum(pr.negative for pr in pattern_reports)
        overall_acc = total_positive / max(total_positive + total_negative, 1)
        if overall_acc < 0.7 and total_positive + total_negative >= 10:
            recs.append(
                f"OVERALL ACCURACY: {overall_acc:.0%} across all patterns. "
                f"Consider reviewing the ranking algorithm in ranker.py."
            )

        return recs

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 7: SAVE REPORT
    # ═══════════════════════════════════════════════════════════════════════════

    def _save_report(self, report: EvaluationReport):
        """Save the weekly report to disk as JSON."""
        os.makedirs(REPORTS_DIR, exist_ok=True)

        filename = f"report_{report.week}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(REPORTS_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        logger.info("Report saved: %s", filepath)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT — run directly with: python -m src.intel.evaluator
# ═══════════════════════════════════════════════════════════════════════════════

# ── Default schedule: 1 week in seconds ──
_ONE_WEEK = 7 * 24 * 60 * 60  # 604 800


def _parse_cli_args() -> tuple[int | None, int]:
    """Parse command-line arguments.

    Returns:
        (loop_interval_seconds | None, lookback_days)

    Flags:
        --loop [SECONDS]   Run repeatedly.  Omit SECONDS to use
                           EVAL_SCHEDULE_SECONDS from config.py
                           (default 604 800 = 1 week).
        --lookback DAYS    How many days of feedback to analyze
                           (default 7).
    """
    import sys
    from src.config import EVAL_SCHEDULE_SECONDS

    args = sys.argv[1:]
    loop_interval: int | None = None
    lookback_days: int = 7

    if "--loop" in args:
        idx = args.index("--loop")
        # Next arg is either a number or another flag (or missing)
        if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
            loop_interval = int(args[idx + 1])
        else:
            loop_interval = EVAL_SCHEDULE_SECONDS

    if "--lookback" in args:
        idx = args.index("--lookback")
        if idx + 1 < len(args):
            lookback_days = int(args[idx + 1])

    return loop_interval, lookback_days


async def main():
    """Entry point — run from CLI or cron.

    Usage:
        python -m src.intel.evaluator                  # one-shot
        python -m src.intel.evaluator --loop            # weekly loop (production)
        python -m src.intel.evaluator --loop 3600       # custom interval
        python -m src.intel.evaluator --lookback 14     # 14-day analysis window
    """
    # ── Configure logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    loop_interval, lookback_days = _parse_cli_args()

    if loop_interval is not None:
        logger.info(
            "Starting evaluation loop — interval=%ds, lookback=%dd",
            loop_interval, lookback_days,
        )
    else:
        logger.info("Running one-shot evaluation — lookback=%dd", lookback_days)

    dao = create_dao()
    await dao._init_db()

    while True:
        try:
            evaluator = Evaluator(dao)
            report = await evaluator.run(lookback_days=lookback_days)

            logger.info(
                "SUMMARY  queries=%d  feedback=%d  changes=%d  recommendations=%d",
                report.total_queries,
                report.total_with_feedback,
                len(report.config_changes),
                len(report.recommendations),
            )

            for c in report.config_changes:
                logger.info("  Change: %s", c)

        except Exception:
            logger.exception("Evaluation loop encountered an error")

        if loop_interval is None:
            break  # one-shot mode

        logger.info("Next evaluation in %ds …", loop_interval)
        await asyncio.sleep(loop_interval)


if __name__ == "__main__":
    asyncio.run(main())
