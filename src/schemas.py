import operator
import sys
from typing import List, Dict

# Python 3.9+ has Annotated and TypedDict in typing, older versions need typing_extensions
if sys.version_info >= (3, 9):
    from typing import Annotated, TypedDict
else:
    from typing_extensions import Annotated, TypedDict

from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    """Structured output for LLM-as-a-Judge evaluation."""
    completeness: int = Field(ge=1, le=5, description="1-5: Did it answer all parts of the prompt?")
    factuality: int = Field(ge=1, le=5, description="1-5: Are claims grounded and verifiable?")
    coherence: int = Field(ge=1, le=5, description="1-5: Is the answer well-structured and clear?")
    reasoning: str = Field(description="Brief explanation of the scores")


class SubQuery(BaseModel):
    """Defines a single step in the research plan (DAG node)."""
    id: int
    question: str = Field(description="The specific search query")
    dependencies: List[int] = Field(default_factory=list, description="IDs of steps that must complete first")
    reasoning: str = Field(default="", description="Why this step is needed")

class ResearchPlan(BaseModel):
    """The collection of sub-queries forming the DAG."""
    steps: List[SubQuery]

class Assessment(BaseModel):
    """The Critic's structured output."""
    is_sufficient: bool = Field(description="True if the draft answers the query completely")
    feedback: str = Field(description="Critique of the current answer")
    new_sub_questions: List[str] = Field(default_factory=list, description="New questions to fill gaps if insufficient")

class AgentState(TypedDict):
    """The Graph State."""
    original_query: str
    
    # Annotated with operator.add to allow appending new steps dynamically
    plan: Annotated[List[SubQuery], operator.add]
    
    # Annotated with operator.ior to allow merging result dicts
    results: Annotated[Dict[int, str], operator.ior] 
    
    draft_answer: str
    critique_count: int
    messages: Annotated[List[str], operator.add]