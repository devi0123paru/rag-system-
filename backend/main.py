"""
main.py — FastAPI Application Entry Point
Includes: standard query, streaming, file upload, incident report, session memory
"""

import os
import uuid
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
load_dotenv()

from models import PatientQuery, RAGResponse, HealthStatus, StreamQuery
from rag_engine import rag_system
from memory import memory_store
from scoring import compute_scores

# ════════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="🚑 EMS Dispatch AI — v2",
    description="""
    Emergency Ambulance RAG System with:
    - Criticality triage (CRITICAL/HIGH/MODERATE/LOW)
    - Medical scoring (Shock Index, NEWS2)
    - Hospital routing recommendations
    - Streaming responses (SSE)
    - Conversation memory (multi-turn)
    - PDF protocol ingestion
    - Incident report generation
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_startup_time = time.time()

# ════════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    persist_path = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    if persist_path.exists() and any(persist_path.iterdir()):
        rag_system.load_index()
    else:
        rag_system.build_index()
    print("✅ EMS RAG API ready")

# ════════════════════════════════════════════════════════════════════════════════
# CORE ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthStatus)
async def health():
    """System health and status."""
    return HealthStatus(
        status="operational",
        service="EMS Dispatch AI v2",
        version="2.0.0",
        index_loaded=rag_system.is_loaded,
        document_count=rag_system._doc_count,
        uptime_seconds=round(rag_system.uptime, 1),
    )

@app.post("/query", response_model=RAGResponse)
async def query(patient_query: PatientQuery):
    """
    Main RAG query endpoint.
    Submit incident details → get criticality, protocol, hospital recommendation.
    """
    try:
        return rag_system.query(patient_query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# ════════════════════════════════════════════════════════════════════════════════
# STREAMING ENDPOINT (NEW)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/query/stream")
async def query_stream(stream_query: StreamQuery):
    """
    Server-Sent Events streaming endpoint.
    Returns tokens in real-time — no waiting for complete response.
   
    Usage:
    const es = new EventSource('/query/stream');
    es.onmessage = (e) => { if (e.data === '[DONE]') es.close(); else updateUI(e.data); }
    """
    async def event_generator():
        async for chunk in rag_system.stream_query(
            query_text=stream_query.query,
            session_id=stream_query.session_id,
            vitals=stream_query.vitals,
        ):
            yield chunk

    return EventSourceResponse(event_generator())

# ════════════════════════════════════════════════════════════════════════════════
# SCORES ENDPOINT (NEW)
# ════════════════════════════════════════════════════════════════════════════════

class ScoreRequest(BaseModel):
    vitals: dict
    age: Optional[int] = None

@app.post("/scores")
async def calculate_scores(req: ScoreRequest):
    """
    Calculate medical scores from vital signs only.
    No LLM call — instant response.
    Returns: Shock Index, NEWS2, critical vital flags.
    """
    from models import VitalSigns
    vitals = VitalSigns(**req.vitals)
    scores = compute_scores(vitals, req.age)
    return scores

# ════════════════════════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD (NEW)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/upload-protocol")
async def upload_protocol(
    file: UploadFile = File(...),
    category: str = Form(default="protocols")
):
    """
    Upload a .txt or .pdf protocol document.
    It gets immediately indexed into ChromaDB — no restart needed.
    """
    allowed = {".txt", ".pdf"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Only .txt and .pdf files allowed. Got: {ext}")

    content_bytes = await file.read()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content_bytes))
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not content.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        except ImportError:
            raise HTTPException(status_code=400, detail="PDF support not installed. Run: pip install pypdf")
    else:
        content = content_bytes.decode("utf-8", errors="ignore")

    # Save to data directory
    save_path = Path(os.getenv("DATA_DIR", "./data")) / category / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content_bytes)

    # Add to live index
    chunks_added = rag_system.add_document(content, file.filename, category)

    return {
        "status": "indexed",
        "filename": file.filename,
        "category": category,
        "chunks_added": chunks_added,
        "saved_to": str(save_path),
    }

# ════════════════════════════════════════════════════════════════════════════════
# SESSION MEMORY (NEW)
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get conversation history for a session."""
    messages = memory_store.get_history(session_id)
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": [m.model_dump() for m in messages]
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's conversation history."""
    memory_store.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

@app.get("/sessions/count")
async def session_count():
    return {"active_sessions": memory_store.get_session_count()}

# ════════════════════════════════════════════════════════════════════════════════
# INCIDENT REPORT (NEW)
# ════════════════════════════════════════════════════════════════════════════════

class ReportRequest(BaseModel):
    incident_data: dict
    dispatcher_notes: Optional[str] = None

@app.post("/incident-report")
async def generate_report(req: ReportRequest):
    """
    Generate a formatted incident report from query response data.
    Returns a structured JSON report (render as PDF in frontend).
    """
    data = req.incident_data
    report = {
        "report_id": f"RPT-{uuid.uuid4().hex[:8].upper()}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "incident_id": data.get("incident_id", "N/A"),
        "criticality": data.get("criticality", "UNKNOWN"),
        "patient": {
            "age": data.get("patient_age"),
            "sex": data.get("patient_sex"),
            "chief_complaint": data.get("chief_complaint"),
            "mechanism": data.get("mechanism"),
        },
        "vitals": data.get("vitals"),
        "scores": data.get("scores"),
        "assessment": data.get("answer"),
        "immediate_actions": data.get("immediate_actions", []),
        "hospital": data.get("hospital"),
        "protocols": data.get("relevant_protocols", []),
        "sources": data.get("sources", []),
        "dispatcher_notes": req.dispatcher_notes,
        "response_time_ms": data.get("response_time_ms"),
    }
    return report

# ════════════════════════════════════════════════════════════════════════════════
# HOSPITAL & INDEX MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/hospitals")
async def list_hospitals():
    data_path = Path(os.getenv("DATA_DIR", "./data")) / "hospitals" / "hospital_directory.txt"
    if not data_path.exists():
        return {"hospitals": []}
    hospitals = []
    for block in data_path.read_text().split("==="):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if lines and any(k in lines[0].upper() for k in ["HOSPITAL", "MEDICAL", "CENTER", "URGENT"]):
            name = lines[0].strip()
            hosp = {"name": name}
            for line in lines[1:]:
                if line.startswith("Type:"):
                    hosp["type"] = line.replace("Type:", "").strip()
                if "Diversion:" in line:
                    hosp["diversion"] = "YES" in line.upper()
            hospitals.append(hosp)
    return {"hospitals": hospitals, "count": len(hospitals)}

@app.post("/rebuild-index")
async def rebuild_index():
    """Force full index rebuild (use after manually adding files to data/)."""
    try:
        rag_system.build_index()
        return {"status": "success", "message": "Index rebuilt", "doc_count": rag_system._doc_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    print(f"\n🚑 Starting EMS Dispatch AI v2 on http://{host}:{port}")
    print(f"📖 API docs: http://localhost:{port}/docs\n")
    uvicorn.run("main:app", host=host, port=port, reload=True)

