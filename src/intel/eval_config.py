"""
Evaluation Config — Adjustable weights and thresholds
=======================================================
This config controls how the Intelligence Layer scores and ranks causes.
The Evaluator (evaluator.py) updates this config weekly based on user feedback.

The pipeline reads this config at runtime:
  - ranker.py      → uses ranking_weights
  - matcher.py     → uses pattern_confidence_overrides
  - single_rca     → uses fast_path_threshold

Config is stored as JSON on disk so changes persist across restarts.
"""

import json
import logging
import os
from typing import Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Default config file location
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "eval_config.json"
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RankingWeights(BaseModel):
    """Weights used in the ranking formula.

    score = latency * latency_w + error_density * error_w + ...

    These get adjusted by the evaluator when it finds that certain
    factors correlate more strongly with correct RCA predictions.
    """
    latency: float = Field(default=0.30, description="Weight for span latency score")
    error_density: float = Field(default=0.25, description="Weight for error/log density")
    pattern_match: float = Field(default=0.20, description="Weight for pattern match score")
    structural: float = Field(default=0.15, description="Weight for tree position/depth")
    temporal: float = Field(default=0.10, description="Weight for timing correlation")

    def normalize(self):
        """Normalize weights so they sum to 1.0."""
        total = self.latency + self.error_density + self.pattern_match + self.structural + self.temporal
        if total > 0:
            self.latency /= total
            self.error_density /= total
            self.pattern_match /= total
            self.structural /= total
            self.temporal /= total

    def to_dict(self) -> dict:
        return {
            "latency": round(self.latency, 4),
            "error_density": round(self.error_density, 4),
            "pattern_match": round(self.pattern_match, 4),
            "structural": round(self.structural, 4),
            "temporal": round(self.temporal, 4),
        }


class EvalConfig(BaseModel):
    """Full evaluation config — read by the pipeline, written by the evaluator.

    Attributes:
        ranking_weights: How much each factor matters in cause ranking
        pattern_confidence_overrides: Per-pattern trust multipliers (0.0 = disabled, 1.0 = full trust)
        fast_path_threshold: Min pattern confidence to skip LLM (fast path)
        min_samples_for_adjustment: Don't adjust patterns with fewer than this many feedback samples
        accuracy_thresholds: Accuracy levels that trigger actions
        last_evaluation: Timestamp of the last evaluation run
        version: Config version (incremented on each evaluation)
    """
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)

    # Per-pattern trust multipliers.
    # If a pattern has accuracy 25%, set its override to 0.3
    # so the ranker multiplies its confidence by 0.3 (effectively lowering it).
    # Missing = 1.0 (full trust, default behavior)
    pattern_confidence_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Pattern name → confidence multiplier (0.0-1.0)"
    )

    # Fast path: skip LLM if pattern confidence > this threshold
    fast_path_threshold: float = Field(
        default=0.85,
        description="Min confidence to use fast path (skip LLM)"
    )

    # Minimum feedback samples before adjusting a pattern
    min_samples_for_adjustment: int = Field(
        default=5,
        description="Need at least this many feedbacks before adjusting"
    )

    # Accuracy thresholds
    accuracy_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "healthy": 0.80,     # >= 80% accuracy = no changes
            "degraded": 0.60,    # 60-80% = lower confidence
            "critical": 0.40,    # < 40% = severely penalize
            "disabled": 0.20,    # < 20% with 10+ samples = disable
        }
    )

    # Metadata
    last_evaluation: str | None = Field(default=None)
    version: int = Field(default=1)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD / SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(path: str = DEFAULT_CONFIG_PATH) -> EvalConfig:
    """Load config from JSON file. Returns defaults if file doesn't exist.

    Usage:
        config = load_config()
        weight = config.ranking_weights.latency  # 0.30
        override = config.pattern_confidence_overrides.get("n_plus_1", 1.0)
    """
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return EvalConfig(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Could not load %s: %s. Using defaults.", path, e)
            return EvalConfig()
    else:
        return EvalConfig()


def save_config(config: EvalConfig, path: str = DEFAULT_CONFIG_PATH):
    """Save config to JSON file.

    Called by the evaluator after adjusting weights.
    """
    data = config.model_dump()
    # Round floats for readability
    if "ranking_weights" in data:
        for k, v in data["ranking_weights"].items():
            if isinstance(v, float):
                data["ranking_weights"][k] = round(v, 4)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("Saved eval_config v%d to %s", config.version, path)


def get_pattern_override(config: EvalConfig, pattern_name: str) -> float:
    """Get the confidence multiplier for a pattern.

    Returns:
        0.0-1.0 multiplier. 1.0 = full trust (default), 0.0 = disabled.

    Usage in matcher.py or ranker.py:
        override = get_pattern_override(config, "n_plus_1")
        adjusted_confidence = pattern.confidence * override
    """
    return config.pattern_confidence_overrides.get(pattern_name, 1.0)
