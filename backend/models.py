"""
models.py — All Pydantic data models for the EMS RAG System
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════════════════════════════════════════════

class VitalSigns(BaseModel):
    hr: Optional[int] = Field(None, description="Heart rate (bpm)")
    bp: Optional[str] = Field(None, description="Blood pressure e.g. '120/80'")
    spo2: Optional[int] = Field(None, description="Oxygen saturation (%)")
    rr: Optional[int] = Field(None, description="Respiratory rate (/min)")
    gcs: Optional[int] = Field(None, ge=3, le=15, description="Glasgow Coma Scale (3-15)")
    temp: Optional[float] = Field(None, description="Temperature (°C)")
    pain: Optional[int] = Field(None, ge=0, le=10, description="Pain score (0-10)")
    bgl: Optional[float] = Field(None, description="Blood glucose level (mg/dL)")

class PatientQuery(BaseModel):
    """Main query model — sent from dispatcher/frontend"""
    query: str = Field(..., description="Free-text incident description")
    patient_age: Optional[int] = Field(None, ge=0, le=130)
    patient_sex: Optional[str] = Field(None, description="M / F / Unknown")
    chief_complaint: Optional[str] = None
    vitals: Optional[VitalSigns] = None
    location: Optional[str] = None
    mechanism: Optional[str] = Field(None, description="Mechanism of injury or illness onset")
    incident_id: Optional[str] = Field(None, description="Dispatcher-assigned incident ID")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    # NEW: follow-up support
    is_followup: bool = Field(False, description="True if this is a follow-up in an ongoing incident")

class StreamQuery(BaseModel):
    """Simplified model for streaming endpoint"""
    query: str
    session_id: Optional[str] = None
    vitals: Optional[VitalSigns] = None

# ════════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ════════════════════════════════════════════════════════════════════════════════

class MedicalScore(BaseModel):
    """Computed medical scoring results"""
    shock_index: Optional[float] = Field(None, description="HR / Systolic BP")
    shock_class: Optional[str] = Field(None, description="Normal / Mild / Moderate / Severe")
    news2_score: Optional[int] = Field(None, description="National Early Warning Score 2")
    news2_risk: Optional[str] = Field(None, description="Low / Medium / High")
    critical_vitals: list[str] = Field(default_factory=list, description="List of critical vital signs")
    pediatric_adjusted: bool = False

class HospitalRecommendation(BaseModel):
    primary: Optional[str] = None
    primary_reason: Optional[str] = None
    secondary: Optional[str] = None
    diversion_active: bool = False
    estimated_transport_mins: Optional[int] = None

class RAGResponse(BaseModel):
    """Full structured response"""
    # Core answer
    answer: str
    criticality: str = Field(..., description="CRITICAL / HIGH / MODERATE / LOW")
    criticality_reason: str

    # Scores (NEW)
    scores: Optional[MedicalScore] = None

    # Hospital routing
    hospital: HospitalRecommendation = Field(default_factory=HospitalRecommendation)

    # Protocols
    relevant_protocols: list[str] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list, description="Bullet-point immediate actions")

    # Meta
    confidence: str = Field("MEDIUM", description="HIGH / MEDIUM / LOW")
    sources: list[str] = Field(default_factory=list)
    response_time_ms: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    incident_id: Optional[str] = None

class IncidentReport(BaseModel):
    """Full incident report — generated on demand"""
    incident_id: str
    timestamp: str
    patient_age: Optional[int]
    patient_sex: Optional[str]
    chief_complaint: Optional[str]
    mechanism: Optional[str]
    vitals: Optional[VitalSigns]
    scores: Optional[MedicalScore]
    criticality: str
    answer: str
    immediate_actions: list[str]
    hospital: HospitalRecommendation
    relevant_protocols: list[str]
    sources: list[str]
    dispatcher_notes: Optional[str] = None

class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class HealthStatus(BaseModel):
    status: str
    service: str
    version: str
    index_loaded: bool
    document_count: int
    uptime_seconds: float