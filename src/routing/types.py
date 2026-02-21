from enum import Enum 
from typing import Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ChatMode(str, Enum):
    """chat mode determine available agents"""
    CHAT = 'chat'       #Only single_rca+general
    AGENT = 'agent'     

class ChatModel(str,Enum):
    """Available LLM models"""
    GPT_4O = 'gpt-4o'
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_1_MINI = "gpt-4.1-mini"
    AUTO = "auto"

class MessageType(str, Enum):
    """Type of message in chat History"""
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    GITHUB = 'auto'

class ActionType(str,Enum):
    AGENT_CHAT = 'agent_chat'
    GITHUB_CREATE_ISSUE = "github_create_issue"
    GITHUB_CREATE_PR = "github_create_pr"
    STATISTICS = "statistics"

class ActionStatus(str, Enum):
    """Status of an action"""
    PENDING = 'pending'
    SUCCESS = 'success'
    FAILED = "failed"

#Request Models
class ChatRequest(BaseModel):
    """Incoming chat request"""

    message:str
    trace_id:Optional[str] = None
    span_ids:list[str] = []
    chat_id: str
    time: datetime
    model:ChatModel = ChatModel.AUTO
    mode: ChatMode = ChatMode.CHAT
    end_time: Optional[datetime] = None 
    service_name : Optional[str] = None
    

#Router Output

class RouterOutput(BaseModel):
    """Output from agent router"""
    agent_type: Literal['single_rca',"code","general"] = Field(
        discription = "Which agent to use"

        )
    
    reasoning:str = Field(
        description="Brief explanation of routing decision"
    )

#Response Models

class Reference(BaseModel):
    """Reference to evidence in trace."""
    type: str
    number: Optional[int] = None
    span_id: Optional[str] = None
    span_function_name: Optional[str] = None
    log_message: Optional[str] = None
    line_number: Optional[int] = None
    url: Optional[str] = None


class IntelligenceMetadata(BaseModel):
    """Intelligence Layer metadata — attached to every RCA response.

    Lets the UI display:
      - Confidence badge (HIGH / MEDIUM / LOW)
      - Pattern name if one was matched
      - Whether fast path was used (no LLM call needed)
      - How long the Intelligence Layer took
      - Top cause info from ranking
    """
    confidence: str                               # "HIGH", "MEDIUM", "LOW"
    pattern_matched: Optional[str] = None
    fast_path: bool = False
    processing_time_ms: float
    top_cause: Optional[str] = None
    top_cause_score: Optional[float] = None
    causes_found: int = 0


class ChatbotResponse(BaseModel):
    """Response to user"""

    time: datetime
    message: str
    reference: list[Reference] = []
    message_type: MessageType = MessageType.ASSISTANT
    chat_id: str
    action: Optional[dict] = None

    # Evidence synthesis fields (populated by synthesizer.py)
    validation_passed: Optional[bool] = None
    validation_confidence: Optional[float] = None
    validation_notes: list[str] = []
    fallback_used: bool = False

    # Intelligence Layer metadata (populated by response_builder)
    metadata: Optional[IntelligenceMetadata] = None
    




