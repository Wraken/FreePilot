import logging
import os

import uvicorn
from log_config import uvicorn_logger
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from errors import FauxPilotException
from codegen import CodeGen
from typing import Optional, Union
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, constr

logging.config.dictConfig(uvicorn_logger)

ModelType = constr(regex="^(fastertransformer|py-model)$")

class OpenAIinput(BaseModel):
    model: ModelType = "fastertransformer"
    prompt: Optional[str]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, list]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 1.0
    best_of: Optional[int] = 1
    logit_bias: Optional[dict] = None
    user: Optional[str] = ""

codegen = CodeGen()

app = FastAPI(
    title="FauxPilot",
    description="This is an attempt to build a locally hosted version of GitHub Copilot.",
    docs_url="/",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    debug=True,
)

# Used to support copilot.vim
@app.get("/copilot_internal/v2/token")
def get_copilot_token():
    content = {'token': '1', 'expires_at': 2600000000, 'refresh_in': 900}
    return JSONResponse(
        status_code=200,
        content=content
    )

@app.post("/v1/engines/codegen/completions")
@app.post("/v1/engines/copilot-codex/completions")
@app.post("/v1/completions")
async def completions(data: OpenAIinput):
    data = data.dict()
    try:
        content = codegen(data=data)
    except Exception as e:
        raise FauxPilotException(
            message="cpt",
            error_type="invalid_request_error",
            param=None,
            code=None,
        )

    if data.get("stream") is not False:
        return EventSourceResponse(
            content=content,
            status_code=200,
            media_type="text/event-stream"
        )
    else:
        return Response(
            status_code=200,
            content=content,
            media_type="application/json"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, log_level="debug")