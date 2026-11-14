"""FastAPI interface for the MLOps agent.

Endpoints
  GET  /health                       liveness + configuration summary (no secrets)
  GET  /tools                        every tool with its permissions
  POST /chat                         ask the agent (natural language)
  GET  /actions                      state-changing actions awaiting approval
  POST /actions/{action_id}/confirm  approve or reject one pending action

Run: uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

import re
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from agent.config import get_settings
from agent.logging_config import configure_logging, get_logger, new_request_id, set_request_id
from agent.orchestrator import Orchestrator, build_orchestrator
from agent.schemas import ChatRequest, ChatResponse, ConfirmRequest, ConfirmResponse, PendingAction

logger = get_logger("api")
_RID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return build_orchestrator()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()  # fail fast on invalid configuration
    configure_logging(settings.log_level, environment=settings.app_env)
    logger.info("api starting", extra={"event": "startup"})
    yield


OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]

app = FastAPI(title="MLOps AI Agent", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    rid = incoming if _RID_RE.match(incoming) else new_request_id()
    set_request_id(rid)
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/health")
def health(orchestrator: OrchestratorDep) -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.app_env,
        "llm_configured": orchestrator.llm is not None,
        "llm_model": settings.llm_model,
        "tools": len(orchestrator.registry.specs()),
    }


@app.get("/tools")
def tools(orchestrator: OrchestratorDep) -> list[dict]:
    return [
        {
            "name": spec.name,
            "source": spec.source.value,
            "read_only": spec.read_only,
            "requires_confirmation": spec.requires_confirmation,
            "description": spec.description,
            "parameters": spec.parameters,
        }
        for spec in orchestrator.registry.specs()
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request, orchestrator: OrchestratorDep) -> ChatResponse:
    if orchestrator.llm is None:
        raise HTTPException(503, "The language model is not configured (set GEMINI_API_KEY)")
    rid = request.state.request_id
    return await run_in_threadpool(
        orchestrator.chat, body.message, request_id=rid, include_evidence=body.include_evidence
    )


@app.get("/actions", response_model=list[PendingAction])
def pending_actions(orchestrator: OrchestratorDep):
    return orchestrator.pending.list()


@app.post("/actions/{action_id}/confirm", response_model=ConfirmResponse)
async def confirm_action(
    action_id: str, body: ConfirmRequest, orchestrator: OrchestratorDep
) -> ConfirmResponse:
    result = await run_in_threadpool(orchestrator.confirm, action_id, body.approve)
    if result is None:
        raise HTTPException(404, "Unknown or expired action id")
    return result
