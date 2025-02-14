from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from schema.hello import Hello 
from schema.responses import ResponseBase, create_response
import structlog

router = APIRouter()
logger = structlog.stdlib.get_logger()


@router.get("/", response_model=ResponseBase[Hello], status_code=200)
async def say_hello(request: Request):
    try:
        name = request.app.state.name
        res = Hello(message=f"Hello, {name}!")
        await logger.info("Someone said hi",name=name)
        return create_response(data=res)
    except KeyError:
        raise HTTPException(status_code=404, detail="Name not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) )