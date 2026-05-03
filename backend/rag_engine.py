"""
rag_engine.py — Enhanced Emergency Ambulance RAG Engine v2
NEW: Groq LLM, streaming, PDF ingestion, conversation memory, medical scoring
"""

import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

# ─── LangChain ────────────────────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# ─── LLM: Groq (fast, free) OR OpenAI ────────────────────────────────────────
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if GROQ_KEY:
    from langchain_groq import ChatGroq
    _LLM_PROVIDER = "groq"
elif OPENAI_KEY:
    from langchain_openai import ChatOpenAI
    _LLM_PROVIDER = "openai"
else:
    raise EnvironmentError(
        "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in your .env file.\n"
        "Get a free Groq key at: https://console.groq.com"
    )

# ─── PDF support ──────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ─── Internal modules ─────────────────────────────────────────────────────────
from models import PatientQuery, RAGResponse, MedicalScore, HospitalRecommendation
from scoring import compute_scores, get_immediate_actions
from memory import memory_store

# ════════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are MEDAI, an expert Emergency Medical Services AI co-pilot.
You assist dispatchers and first responders with life-or-death decisions.

Your response MUST follow this EXACT format:

CRITICALITY: [CRITICAL|HIGH|MODERATE|LOW] — [one-line clinical reason]

IMMEDIATE ACTIONS:
1. [First action]
2. [Second action]
3. [Continue as needed]

PROTOCOL GUIDANCE:
[Relevant protocol steps from your knowledge base]

HOSPITAL RECOMMENDATION: [Hospital name] — [Why this hospital specifically]

ROUTING NOTES: [Pre-notification language, special instructions]

ADDITIONAL NOTES: [Drug dosages, special considerations, red flags to watch]

Rules:
- Be direct, clinical, and actionable
- Lead with the most time-critical information
- If vitals are critical, say so explicitly
- Base everything on the provided context
- Never guess — if uncertain, say so and give conservative advice
{memory_context}
Knowledge Base Context:
{context}

Incident Query: {question}"""

# ════════════════════════════════════════════════════════════════════════════════
# RAG ENGINE CLASS
# ════════════════════════════════════════════════════════════════════════════════

class AmbulanceRAGSystem:
    def __init__(self):
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain = None
        self._start_time = time.time()
        self._doc_count = 0

        print("⚡ Loading embedding model (sentence-transformers)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print(f"🤖 LLM Provider: {_LLM_PROVIDER.upper()}")
        self.llm = self._init_llm(streaming=False)
        self.streaming_llm = self._init_llm(streaming=True)

    def _init_llm(self, streaming: bool = False):
        if _LLM_PROVIDER == "groq":
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.05,
                max_tokens=2000,
                streaming=streaming,
                groq_api_key=GROQ_KEY,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.05,
                max_tokens=2000,
                streaming=streaming,
            )

    # ── Document Loading ──────────────────────────────────────────────────────

    def _load_documents(self) -> list[Document]:
        documents = []

        for txt_file in self.data_dir.rglob("*.txt"):
            try:
                content = txt_file.read_text(encoding="utf-8")
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": txt_file.name,
                        "category": txt_file.parent.name,
                        "type": "text",
                    }
                ))
            except Exception as e:
                print(f"⚠ Could not load {txt_file.name}: {e}")

        if PDF_SUPPORT:
            for pdf_file in self.data_dir.rglob("*.pdf"):
                try:
                    reader = PdfReader(str(pdf_file))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": pdf_file.name,
                                "category": pdf_file.parent.name,
                                "type": "pdf",
                                "pages": len(reader.pages),
                            }
                        ))
                        print(f"  📄 PDF: {pdf_file.name} ({len(reader.pages)} pages)")
                except Exception as e:
                    print(f"⚠ Could not load PDF {pdf_file.name}: {e}")

        return documents

    # ── Index ─────────────────────────────────────────────────────────────────

    def build_index(self) -> None:
        print("📄 Loading EMS documents...")
        documents = self._load_documents()
        self._doc_count = len(documents)
        print(f"✅ Loaded {self._doc_count} documents")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n===", "\n---", "\n\n", "\n", " "],
            keep_separator=True,
        )
        chunks = splitter.split_documents(documents)
        print(f"📦 Created {len(chunks)} chunks")

        print("🔍 Building vector index (ChromaDB)...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="ems_protocols_v2"
        )
        self.vectorstore.persist()
        print(f"✅ Index saved to {self.persist_dir}")
        self._build_chain()

    def load_index(self) -> None:
        print("📂 Loading existing vector index...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="ems_protocols_v2"
        )
        self._doc_count = self.vectorstore._collection.count()
        self._build_chain()
        print(f"✅ Index loaded ({self._doc_count} chunks)")

    def add_document(self, content: str, source_name: str, category: str = "custom") -> int:
        """Live document addition — no full rebuild needed."""
        if not self.vectorstore:
            raise RuntimeError("Index not loaded.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        doc = Document(
            page_content=content,
            metadata={"source": source_name, "category": category, "type": "uploaded"}
        )
        chunks = splitter.split_documents([doc])
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        self._doc_count += 1
        return len(chunks)

    def _build_chain(self) -> None:
        prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question", "memory_context"]
        )
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.7}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, patient_query: PatientQuery) -> RAGResponse:
        if not self.qa_chain:
            raise RuntimeError("Index not loaded.")

        start = time.time()
        enriched = _build_enriched_query(patient_query)

        memory_ctx = ""
        if patient_query.session_id:
            memory_ctx = memory_store.get_context_string(patient_query.session_id)
            memory_store.add_message(patient_query.session_id, "user", patient_query.query)

        result = self.qa_chain.invoke({
            "query": enriched,
            "memory_context": memory_ctx,
        })

        answer = result["result"]
        source_docs = result.get("source_documents", [])

        if patient_query.session_id:
            memory_store.add_message(patient_query.session_id, "assistant", answer[:500])

        scores = compute_scores(patient_query.vitals, patient_query.patient_age)
        criticality, criticality_reason = _parse_criticality(answer, patient_query.vitals, scores)
        hospital = _parse_hospital_rec(answer)
        protocols = _extract_protocols(source_docs)
        sources = list({doc.metadata.get("source", "Unknown") for doc in source_docs})
        immediate_actions = get_immediate_actions(criticality, patient_query.chief_complaint)

        if scores.critical_vitals and criticality == "CRITICAL":
            immediate_actions = [f"⚠ {v}" for v in scores.critical_vitals] + immediate_actions

        elapsed_ms = int((time.time() - start) * 1000)

        return RAGResponse(
            answer=answer,
            criticality=criticality,
            criticality_reason=criticality_reason,
            scores=scores,
            hospital=hospital,
            relevant_protocols=protocols,
            immediate_actions=immediate_actions[:8],
            confidence=_assess_confidence(source_docs),
            sources=sources,
            response_time_ms=elapsed_ms,
            incident_id=patient_query.incident_id or f"INC-{uuid.uuid4().hex[:6].upper()}",
        )

    async def stream_query(self, query_text: str, session_id: Optional[str] = None, vitals=None) -> AsyncGenerator[str, None]:
        if not self.vectorstore:
            yield "data: ERROR: Index not loaded\n\n"
            return

        retriever = self.vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
        )
        docs = retriever.get_relevant_documents(query_text)
        context = "\n\n".join(doc.page_content for doc in docs)

        memory_ctx = ""
        if session_id:
            memory_ctx = memory_store.get_context_string(session_id)
            memory_store.add_message(session_id, "user", query_text)

        prompt_text = SYSTEM_PROMPT.format(
            context=context,
            question=query_text,
            memory_context=memory_ctx
        )

        full_response = ""
        async for chunk in self.streaming_llm.astream(prompt_text):
            token = chunk.content
            if token:
                full_response += token
                # Escape for SSE
                safe = token.replace("\n", "\\n")
                yield f"data: {safe}\n\n"

        if session_id:
            memory_store.add_message(session_id, "assistant", full_response[:500])

        yield "data: [DONE]\n\n"

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    @property
    def is_loaded(self) -> bool:
        return self.vectorstore is not None

# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _build_enriched_query(q: PatientQuery) -> str:
    parts = [q.query]
    if q.patient_age:
        parts.append(f"Patient: {q.patient_age}yo {q.patient_sex or ''}.")
    if q.chief_complaint:
        parts.append(f"Chief complaint: {q.chief_complaint}.")
    if q.mechanism:
        parts.append(f"Mechanism: {q.mechanism}.")
    if q.vitals:
        v = q.vitals
        vital_parts = []
        if v.hr:   vital_parts.append(f"HR={v.hr}")
        if v.bp:   vital_parts.append(f"BP={v.bp}")
        if v.spo2: vital_parts.append(f"SpO2={v.spo2}%")
        if v.rr:   vital_parts.append(f"RR={v.rr}")
        if v.gcs:  vital_parts.append(f"GCS={v.gcs}")
        if v.temp: vital_parts.append(f"Temp={v.temp}°C")
        if v.bgl:  vital_parts.append(f"BGL={v.bgl}")
        if vital_parts:
            parts.append(f"Vitals: {', '.join(vital_parts)}.")
    if q.location:
        parts.append(f"Location: {q.location}.")
    parts.append("Assess criticality, provide protocol, recommend hospital.")
    return " ".join(parts)

def _parse_criticality(answer: str, vitals, scores: MedicalScore) -> tuple[str, str]:
    if scores and scores.critical_vitals:
        return "CRITICAL", f"Critical vital signs: {scores.critical_vitals[0]}"
    if scores and scores.shock_index and scores.shock_index >= 1.2:
        return "CRITICAL", f"Severe shock (SI={scores.shock_index})"

    upper = answer.upper()
    for level in ["CRITICAL", "HIGH", "MODERATE", "LOW"]:
        if f"CRITICALITY: {level}" in upper:
            return level, _extract_reason_text(answer, level)

    return "UNKNOWN", "Could not determine criticality"

def _extract_reason_text(answer: str, level: str) -> str:
    idx = answer.upper().find(f"CRITICALITY: {level}")
    if idx >= 0:
        line = answer[idx:idx+250].split("\n")[0]
        parts = line.split("—")
        if len(parts) > 1:
            return parts[1].strip()[:150]
    return f"Classified as {level}"

def _parse_hospital_rec(answer: str) -> HospitalRecommendation:
    rec = HospitalRecommendation()
    for line in answer.split("\n"):
        if "HOSPITAL RECOMMENDATION:" in line.upper():
            content = line.split(":", 1)[-1].replace("**", "").strip()
            if " — " in content:
                parts = content.split(" — ", 1)
                rec.primary = parts[0].strip()
                rec.primary_reason = parts[1].strip()
            else:
                rec.primary = content[:100]
        if "ROUTING NOTES:" in line.upper():
            note = line.split(":", 1)[-1].strip()
            rec.primary_reason = (rec.primary_reason or "") + " | " + note
    return rec

def _extract_protocols(docs) -> list[str]:
    protocols = set()
    for doc in docs:
        matches = re.findall(r"PROTOCOL\s+[A-Z]-\d+", doc.page_content)
        protocols.update(matches)
    return list(protocols)[:6]

def _assess_confidence(docs) -> str:
    n = len(docs)
    if n >= 5: return "HIGH"
    if n >= 3: return "MEDIUM"
    return "LOW"

# Global singleton
rag_system = AmbulanceRAGSystem()

