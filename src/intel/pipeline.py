import time
from src.intel.types import (
    IntelligenceResult, IntelligenceConfig, ClassifiedSpanNode,
    RankingConfig,
)
from src.intel.Classifier import classify_spans
from src.intel.supressor import suppress_noise
from src.intel.locator import locate_failures
from src.intel.matcher import match_patterns
from src.intel.Ranker import rank_causes
from src.intel.compressor import compress_context
from src.intel.eval_config import EvalConfig, load_config
from src.config import EVAL_CONFIG_PATH
from src.context.model import SpanNode


def _build_intel_config(eval_cfg: EvalConfig) -> IntelligenceConfig:
    """Merge EvalConfig feedback-adjusted values into IntelligenceConfig.

    Maps eval_config.ranking_weights → RankingConfig used by ranker.py,
    and eval_config.fast_path_threshold → IntelligenceConfig.fast_path_threshold.
    """
    w = eval_cfg.ranking_weights
    ranking = RankingConfig(
        error_density_weight=w.error_density,
        latency_anomaly_weight=w.latency,
        pattern_match_weight=w.pattern_match,
        depth_weight=w.structural,
        temporal_correlation_weight=w.temporal,
        # span_type_weight stays at default — not tuned by evaluator
    )
    return IntelligenceConfig(
        ranking=ranking,
        fast_path_threshold=eval_cfg.fast_path_threshold,
    )


class IntelligencePipeline:
    """Main orchestrator for the intelligence layer.

    On init, loads eval_config.json (written by the Evaluator) and merges
    its feedback-adjusted weights/thresholds into IntelligenceConfig.
    This means:
      - ranker.py  sees adjusted ranking_weights
      - matcher.py sees pattern_confidence_overrides
      - fast-path  uses the adjusted threshold
    """

    def __init__(self, config: IntelligenceConfig = None):
        # Load evaluation config (feedback loop)
        self.eval_config: EvalConfig = load_config(EVAL_CONFIG_PATH)

        # If caller supplied an explicit config, use it;
        # otherwise build one from eval_config.
        if config is not None:
            self.config = config
        else:
            self.config = _build_intel_config(self.eval_config)
    
    async def process(
        self,
        tree: SpanNode,
        user_query: str,
    ) -> IntelligenceResult:
        """Main entry point for intelligence processing."""
        start_time = time.time()
        
        # Step 1: Semantic Classification
        classified_tree = classify_spans(tree)
        
        # Step 2: Noise Suppression
        pruned_tree, suppression_stats = suppress_noise(
            classified_tree,
            config=self.config.suppression
        )
        
        # Step 3: Failure Locality Detection
        failure_report = locate_failures(pruned_tree)
        
        # Step 4: Pattern Matching
        pattern_matches = match_patterns(
            pruned_tree,
            failure_report,
            user_query,
            pattern_confidence_overrides=self.eval_config.pattern_confidence_overrides,
        )
        
        # Step 5: Cause Ranking
        ranked_causes = rank_causes(
            pruned_tree,
            failure_report,
            pattern_matches,
            config=self.config.ranking
        )
        
        # Step 6: Context Compression
        compressed_context = compress_context(
            pruned_tree,
            ranked_causes,
            token_budget=self.config.token_budget
        )
        
        # Determine if fast path is available
        # Pattern confidence alone is sufficient — the pattern matcher already
        # validates match quality (e.g., N+1 needs 3+ similar queries).
        # NOTE: We don't require ranked_causes.confidence == HIGH because
        # the Ranker scores individual spans, but patterns like N+1 are
        # aggregate issues where no single span scores high on its own.
        fast_path = (
            len(pattern_matches) > 0 and
            pattern_matches[0].confidence > self.config.fast_path_threshold
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return IntelligenceResult(
            classified_tree=classified_tree,
            suppression_stats=suppression_stats,
            failure_report=failure_report,
            pattern_matches=pattern_matches,
            ranked_causes=ranked_causes,
            compressed_context=compressed_context,
            fast_path_available=fast_path,
            processing_time_ms=processing_time
        )