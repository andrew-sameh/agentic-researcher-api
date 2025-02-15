from pydantic import BaseModel
from typing import Optional, List

class ResearchRequest(BaseModel):
    query: str
    context: Optional[str] = None
    feedback: Optional[str] = None

class ResearchResponse(BaseModel):
    answer: str
    citations: List[str]
    requires_feedback: bool

