import time
from src.intel.types import (
    IntelligenceResult, IntelligenceConfig, ClassifiedSpanNode
)
from src.intel.Classifier import classify_spans
from src.intel.supressor import suppress_noise
from src.intel.locator import locate_failures
from src.intel.matcher import match_patterns
from src.intel.Ranker import rank_causes
from src.intel.compressor import compress_context
from src.context.model import SpanNode

class IntelligencePipeline:
    """Main orchestrator for the intelligence layer."""
    
    def __init__(self, config: IntelligenceConfig = None):
        self.config = config or IntelligenceConfig()
    
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
            user_query
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
        fast_path = (
            len(pattern_matches) > 0 and
            pattern_matches[0].confidence > self.config.fast_path_threshold and
            ranked_causes.confidence_level.value == "high"
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