from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from schema.health import HealthCheck
from schema.responses import ResponseBase, create_response
import structlog

router = APIRouter()
logger = structlog.stdlib.get_logger()


@router.get("/", response_model=ResponseBase[HealthCheck], status_code=200)
async def health_check(request: Request):
    try:
        name = request.app.state.name
        res = HealthCheck(message=f"Hello, {name}, I am alive!")
        await logger.info("Someone checked the health of the API",name=name)
        return create_response(data=res)
    except KeyError:
        raise HTTPException(status_code=404, detail="Name not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) )