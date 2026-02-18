from abc import ABC, abstractmethod
from src.intel.types import ClassifiedSpanNode, FailureReport, PatternMatch
class FailurePattern(ABC):
    """Base class for failure patterns."""
    
    @property
    @abstractmethod
    def pattern_id(self) -> str:
        pass
    
    @property
    @abstractmethod
    def pattern_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def pattern_category(self) -> str:
        pass
    
    @abstractmethod
    def match(
        self,
        tree: ClassifiedSpanNode,
        failure_report: FailureReport
    ) -> PatternMatch | None:
        """Return PatternMatch if pattern matches, None otherwise."""
        pass