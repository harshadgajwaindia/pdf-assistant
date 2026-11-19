import logging
import os
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.rag_service import RagService

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mini RAG PDF Assistant API", version="1.0.0")
_service: RagService | None = None
_service_lock = Lock()
MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)


def get_service() -> RagService:
    global _service
    with _service_lock:
        if _service is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=503,
                    detail="Set GOOGLE_API_KEY in the environment before using the assistant.",
                )
            _service = RagService(api_key=api_key)
        return _service


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdfs(
    session_id: str = Form(..., min_length=1, max_length=128),
    files: list[UploadFile] = File(...),
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one PDF to upload.")

    uploads: list[tuple[str, bytes]] = []
    for file in files:
        filename = file.filename or "uploaded.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{filename} is not a PDF file.")
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{filename} is empty.")
        if len(content) > MAX_PDF_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the 20 MB per-file limit.",
            )
        uploads.append((filename, content))

    service = get_service()
    try:
        result = await run_in_threadpool(service.add_pdfs, session_id, uploads)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF processing failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {exc}") from exc
    return result


@app.post("/ask")
async def ask_question(request: AskRequest) -> dict[str, object]:
    service = get_service()
    try:
        result = await run_in_threadpool(
            service.answer_question, request.session_id, request.question
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Question answering failed for session %s", request.session_id)
        raise HTTPException(status_code=500, detail=f"Could not answer the question: {exc}") from exc
    return result
