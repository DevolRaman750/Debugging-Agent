from enum import Enum
from typing import NewType, Optional
from pydantic import BaseModel
from datetime import datetime

#ENUMS
class SpanType(str, Enum):

    LLM_CALL = "llm_call"
    DB_QUERY = "db_query"
    HTTP_REQUEST = "http_request"
    HTTP_HANDLER = "http_handler"
    QUEUE_PUBLISH = "queue_publish"
    QUEUE_CONSUME = "queue_consume"
    CACHE_OP = "cache_op"
    FILE_IO = "file_io"
    RETRY_LOOP = "retry_loop"
    HEALTH_CHECK = "health_check"
    INSTRUMENTATION = "instrumentation"
    BUSINESS_LOGIC = "business_logic"
    UNKNOWN = "unknown"

class FailureType(str, Enum):
    """Type of failure detected."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    VALIDATION_ERROR = "validation_error"
    AUTH_FAILURE = "auth_failure"
    NOT_FOUND = "not_found"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"

class ConfidenceLevel(str, Enum):

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


#Classifier Output
class ClassifiedSpanNode(BaseModel):
    """SpanNode with semantic classification added."""

    span_id: str
    func_full_name: str
    span_latency:float
    span_utc_start_time: datetime
    span_utc_end_time: datetime
    logs:list
    children_spans:list

    span_type: SpanType
    classification_confidence: float  # 0.0 to 1.0


#Suppressor Output
class SuppressionStats(BaseModel):
    """Statistics from noise suppression."""

    original_span_count: int
    remaining_span_count: int
    suppressed_count: int
    collapsed_count:int
    compression_ratio:float
    
# FAILURE LOCATOR OUTPUT

class FailureSpan(BaseModel):
    """A span identified as containing a failure."""

    span_id:str
    span_function:str
    failure_type:FailureType
    error_messages:list[str]
    is_root_cause:bool
    depth_in_tree:int

class FailureChain(BaseModel):
    """Chain of spans from root cause to symptom"""

    spans:list[str]
    propagation_path:str

class FailureReport(BaseModel):
    """Complete failure analysis."""
    failure_spans:list[FailureSpan]
    failure_chains:list[FailureChain]
    root_cause_candidates:list[str]
    has_failures:bool

#pattern matcher output
class PatternMatch(BaseModel):
     """A matched failure pattern."""

     pattern_id:str
     pattern_name:str
     pattern_category:str
     confidence:float
     matched_spans: list[str]
     matched_evidence: list[str]
     explanation:str
     recommended_fix:str

class ScoringFactor(BaseModel):
    factor_name:str
    factor_value:float
    weighted_contribution:float

class Evidence(BaseModel):
    """Evidence supporting a  cause"""
    span_id:str
    evidence_type:str
    description:str
    timestamp:Optional[datetime]=None


class RankedCause(BaseModel):
    """A ranked candidate cause."""
    span_id: str
    span_function: str
    score: float  # 0.0 - 1.0
    rank: int  # 1 = top cause
    contributing_factors: list[ScoringFactor]
    evidence: list[Evidence]

class RankedCauses(BaseModel):
    """Complete ranking result."""
    causes: list[RankedCause]  # sorted by score desc
    confidence_level: ConfidenceLevel
    top_cause_score: float
    score_gap: float  # gap between #1 and #2

#Compressor Output
class CompressedContext(BaseModel):
    """Compressed context for LLM consumption."""
    compressed_tree: dict  # JSON-serializable tree
    token_count: int
    original_token_count: int
    compression_ratio: float
    preserved_evidence: list[Evidence]
    truncation_notes: list[str]

#Main PipeLine Output
class IntelligenceResult(BaseModel):
    """Complete output from intelligence pipeline."""
    classified_tree: ClassifiedSpanNode
    suppression_stats: SuppressionStats
    failure_report: FailureReport
    pattern_matches: list[PatternMatch]
    ranked_causes: RankedCauses
    compressed_context: CompressedContext
    fast_path_available: bool
    processing_time_ms: float

#Configurations 
class SuppressionConfig(BaseModel):
    """Configuration for noise suppression."""
    suppress_health_checks: bool = True
    suppress_instrumentation: bool = True
    min_duration_ms: float = 1.0
    collapse_repeated: bool = True
    collapse_threshold: int = 3

class RankingConfig(BaseModel):
    """Configuration for cause ranking."""
    error_density_weight: float = 0.25
    latency_anomaly_weight: float = 0.20
    temporal_correlation_weight: float = 0.20
    pattern_match_weight: float = 0.15
    depth_weight: float = 0.10
    span_type_weight: float = 0.10

class IntelligenceConfig(BaseModel):
    """Main configuration for intelligence pipeline."""
    suppression: SuppressionConfig = SuppressionConfig()
    ranking: RankingConfig = RankingConfig()
    token_budget: int = 5000
    fast_path_threshold: float = 0.8



        




    






