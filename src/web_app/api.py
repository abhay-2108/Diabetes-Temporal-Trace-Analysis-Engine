"""
DiaTrace.AI — FastAPI Backend
Exposes REST endpoints for the React frontend to query patient data
and trigger the multi-agent clinical analysis pipeline.
"""
import os
import sys
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.database.db_handler import DB_PATH, get_patient_timeline
from src.agents.orchestrator_agent import (
    PatientClinicalState,
    PatientOrchestrator,
    analyze_timeline_biomarkers_mathematically,
)
from src.vector_db.ada_vector_store import LocalADAVectorStore

# ── App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="DiaTrace.AI Clinical API",
    description="Multi-agent diabetes clinical decision support system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load the Vector DB once on first request
_vector_db: Optional[LocalADAVectorStore] = None

def get_vector_db() -> LocalADAVectorStore:
    global _vector_db
    if _vector_db is None:
        _vector_db = LocalADAVectorStore()
    return _vector_db


# ── Request / Response Models ────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    patient_nbr: int


class AnalyzeCustomRequest(BaseModel):
    notes: str


class PatientSummary(BaseModel):
    patient_nbr: int
    race: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    visit_count: int = 0


class AnalysisResponse(BaseModel):
    patient_nbr: int
    demographics: Dict[str, Any]
    visit_count: int
    biomarkers: Dict[str, Any]
    complications: List[str]
    ada_clashes: List[str]
    evidence_chain: List[str]
    recommendations: List[str]
    raw_report: str


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/api/patients", response_model=List[PatientSummary])
def list_patients(limit: int = 50):
    """Return a list of patient IDs from the EHR database."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="EHR database not found.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT p.patient_nbr, p.race, p.gender, p.age, "
            "COUNT(e.encounter_id) AS visit_count "
            "FROM dim_patients p "
            "LEFT JOIN fact_encounters e ON p.patient_nbr = e.patient_nbr "
            "GROUP BY p.patient_nbr "
            "ORDER BY visit_count DESC "
            f"LIMIT {limit}"
        ).fetchall()
        return [
            PatientSummary(
                patient_nbr=r["patient_nbr"],
                race=r["race"],
                gender=r["gender"],
                age=r["age"],
                visit_count=r["visit_count"],
            )
            for r in rows
        ]
    finally:
        conn.close()


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze_patient(req: AnalyzeRequest):
    """Run the full multi-agent pipeline on a patient and return structured results."""
    raw_record = get_patient_timeline(req.patient_nbr)
    if not raw_record:
        raise HTTPException(status_code=404, detail=f"Patient {req.patient_nbr} not found.")

    demographics = raw_record.get("demographics", {})
    raw_timeline = raw_record.get("timeline", [])

    # Mathematical analysis (deterministic)
    math_results = analyze_timeline_biomarkers_mathematically(raw_timeline)

    # Build state
    state = PatientClinicalState(
        patient_nbr=req.patient_nbr,
        demographics=demographics,
        raw_timeline=raw_timeline,
    )

    # Determine if we should bypass Mock and run the real LLM pipeline
    use_mock = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
    
    vdb = get_vector_db()
    orchestrator = PatientOrchestrator(state, vdb, use_mock_llm=use_mock)
    raw_report = orchestrator.run_orchestration()

    # Extract eGFR values for biomarker info
    egfr_values = [v.get("eGFR") for v in raw_timeline if isinstance(v.get("eGFR"), (int, float))]
    a1c_values = [v.get("A1Cresult") for v in raw_timeline if v.get("A1Cresult") and v.get("A1Cresult") not in ("None", "Not Tested")]

    # Parse recommendations from the report (isolate Section 5)
    recommendations = []
    parts = raw_report.split("#### 5.")
    if len(parts) > 1:
        recommendations_section = parts[1]
        for line in recommendations_section.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ")):
                recommendations.append(stripped.lstrip("0123456789. "))
    else:
        for line in raw_report.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("1. **", "2. **", "3. **", "4. **")) and "EVIDENCE" not in stripped:
                recommendations.append(stripped.lstrip("0123456789. "))


    return AnalysisResponse(
        patient_nbr=req.patient_nbr,
        demographics={
            "age": demographics.get("age", "Unknown"),
            "race": demographics.get("race", "Unknown"),
            "gender": demographics.get("gender", "Unknown"),
        },
        visit_count=len(raw_timeline),
        biomarkers={
            "egfr_slope": round(math_results["egfr_slope"], 2),
            "latest_egfr": egfr_values[-1] if egfr_values else None,
            "egfr_trajectory": egfr_values,
            "a1c_values": a1c_values,
            "decline_assessment": math_results["decline_assessment"],
        },
        complications=math_results["complications"],
        ada_clashes=math_results["ada_clashes"],
        evidence_chain=math_results["evidence_chain"],
        recommendations=recommendations,
        raw_report=raw_report,
    )


@app.post("/api/analyze_custom", response_model=AnalysisResponse)
def analyze_custom_note(req: AnalyzeCustomRequest):
    """Run the multi-agent pipeline on raw, custom unstructured clinical notes."""
    if not req.notes.strip():
        raise HTTPException(status_code=400, detail="Clinical note text cannot be empty.")

    # Create a virtual, single-encounter raw timeline
    raw_timeline = [{
        "visit_index": 1,
        "encounter_id": 999999,
        "eGFR": 45,  # Default safe placeholder for custom assessment
        "A1Cresult": "None",
        "clinician_note": req.notes,
        "medication_adjustments": []
    }]
    
    # Mathematical analysis
    math_results = analyze_timeline_biomarkers_mathematically(raw_timeline)

    state = PatientClinicalState(
        patient_nbr=9999,
        demographics={"age": "Unknown", "race": "Unknown", "gender": "Unknown"},
        raw_timeline=raw_timeline,
    )

    use_mock = os.getenv("USE_MOCK_LLM", "true").lower() == "true"
    vdb = get_vector_db()
    
    # Run orchestrator
    orchestrator = PatientOrchestrator(state, vdb, use_mock_llm=use_mock)
    raw_report = orchestrator.run_orchestration()

    # Isolate biomarker info
    egfr_values = [45]
    a1c_values = []

    # Parse recommendations
    recommendations = []
    parts = raw_report.split("#### 5.")
    if len(parts) > 1:
        recommendations_section = parts[1]
        for line in recommendations_section.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ")):
                recommendations.append(stripped.lstrip("0123456789. "))
    else:
        for line in raw_report.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("1. **", "2. **", "3. **", "4. **")) and "EVIDENCE" not in stripped:
                recommendations.append(stripped.lstrip("0123456789. "))

    return AnalysisResponse(
        patient_nbr=9999,
        demographics={
            "age": "Unknown",
            "race": "Unknown",
            "gender": "Unknown",
        },
        visit_count=1,
        biomarkers={
            "egfr_slope": 0.0,
            "latest_egfr": 45,
            "egfr_trajectory": egfr_values,
            "a1c_values": a1c_values,
            "decline_assessment": "Assessing custom clinical note text.",
        },
        complications=math_results["complications"],
        ada_clashes=math_results["ada_clashes"],
        evidence_chain=math_results["evidence_chain"],
        recommendations=recommendations,
        raw_report=raw_report,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web_app.api:app", host="127.0.0.1", port=8000, reload=True)

