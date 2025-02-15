from fastapi import APIRouter, HTTPException
from agent.workflow import ResearchAgent
from schema.research import ResearchRequest, ResearchResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
agent = ResearchAgent()

@router.post("/query", response_model=ResearchResponse)
async def research_query(request: ResearchRequest):
    try:
        pass
    except Exception as e:
        logger.error(f"Error processing research query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 