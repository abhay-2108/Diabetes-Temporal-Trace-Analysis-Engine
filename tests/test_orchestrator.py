import os
import json
import pytest
import sqlite3
from typing import Dict, Any

from src.database.db_handler import DB_PATH, get_patient_timeline
from src.vector_db.ada_vector_store import LocalADAVectorStore
from src.agents.orchestrator_agent import (
    PatientClinicalState,
    analyze_timeline_biomarkers_mathematically,
    MockClinicalLLM,
    PatientOrchestrator
)

def get_any_valid_patient_nbr() -> int:
    """Helper to query a valid patient from dim_patients in SQLite."""
    if not os.path.exists(DB_PATH):
        return 1000
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT patient_nbr FROM dim_patients LIMIT 1").fetchone()
        return row[0] if row else 1000
    finally:
        conn.close()


def test_sqlite_timeline_extraction():
    """Verify that get_patient_timeline extracts valid Demographics and chronological encounters."""
    if not os.path.exists(DB_PATH):
        pytest.skip("EHR SQLite database is missing. Skipping test.")
        
    patient_nbr = get_any_valid_patient_nbr()
    timeline_rec = get_patient_timeline(patient_nbr)
    
    assert timeline_rec is not None, "Timeline record is empty"
    assert "patient_nbr" in timeline_rec, "Missing patient_nbr in record"
    assert "demographics" in timeline_rec, "Missing demographics dimension"
    assert "timeline" in timeline_rec, "Missing chronological timeline fact list"
    
    demographics = timeline_rec["demographics"]
    assert "race" in demographics
    assert "gender" in demographics
    assert "age" in demographics
    
    timeline = timeline_rec["timeline"]
    assert isinstance(timeline, list)
    if len(timeline) > 0:
        first_visit = timeline[0]
        assert "visit_index" in first_visit
        assert "eGFR" in first_visit
        assert "A1Cresult" in first_visit
        assert "clinician_note" in first_visit


def test_ada_vector_store():
    """Verify that LocalADAVectorStore initializes and successfully performs guideline searches."""
    vdb = LocalADAVectorStore()
    assert vdb is not None
    
    # Query for Metformin contraindications
    results = vdb.query("metformin contraindicated eGFR < 30", n_results=1)
    assert len(results) > 0, "No guidelines returned"
    
    # Assert section mapping exists
    match = results[0]
    assert "section" in match
    assert "text" in match
    assert "id" in match


def test_mathematical_biomarker_slope_calculator():
    """Verify that eGFR slope and microvascular complication checks calculate correctly."""
    # Construct a synthetic timeline representing rapidly declining eGFR and high HbA1c
    synthetic_timeline = [
        {
            "visit_index": 1,
            "eGFR": 90,
            "A1Cresult": ">8",
            "clinician_note": "Normal baseline check. Active Metformin.",
            "medication_adjustments": [{"drug_name": "metformin", "dosage_status": "Steady"}]
        },
        {
            "visit_index": 2,
            "eGFR": 75,
            "A1Cresult": ">8",
            "clinician_note": "Mild bilateral toe tingling noted. Retinal blurred vision reporting.",
            "medication_adjustments": [{"drug_name": "metformin", "dosage_status": "Steady"}]
        },
        {
            "visit_index": 3,
            "eGFR": 60,
            "A1Cresult": ">8",
            "clinician_note": "CKD progression, tingling persists.",
            "medication_adjustments": [{"drug_name": "metformin", "dosage_status": "Steady"}]
        },
        {
            "visit_index": 4,
            "eGFR": 28,  # Under 30, triggers Metformin contraindication!
            "A1Cresult": ">8",
            "clinician_note": "Severe renal clearance degradation.",
            "medication_adjustments": [{"drug_name": "metformin", "dosage_status": "Steady"}]
        }
    ]
    
    # Total change = 28 - 90 = -62 units
    # Total years = 3 intervals * 0.5 years/interval = 1.5 years
    # Expected slope = -62 / 1.5 = -41.33 mL/min/1.73m²/year
    
    analysis = analyze_timeline_biomarkers_mathematically(synthetic_timeline)
    
    assert analysis["egfr_slope"] == pytest.approx(-41.333333333333336)
    assert "Diabetic Kidney Disease (Rapid Progression)" in analysis["complications"]
    assert "Severe Chronic Kidney Disease (Stage 4)" in analysis["complications"]
    assert "Diabetic Peripheral Neuropathy" in analysis["complications"]
    assert "Diabetic Retinopathy" in analysis["complications"]
    
    # Ensure Metformin contraindication is flagged
    assert any("contraindication" in c.lower() for c in analysis["ada_clashes"])
    # Ensure clinical inertia is flagged (all 4 A1cs were high)
    assert any("clinical inertia" in c.lower() for c in analysis["ada_clashes"])


def test_crewai_multi_agent_pipeline():
    """Verify that build_and_run_crewai_orchestrator successfully coordinates agents and yields audit reports."""
    if not os.path.exists(DB_PATH):
        pytest.skip("EHR SQLite database is missing. Skipping test.")
        
    patient_nbr = get_any_valid_patient_nbr()
    raw_record = get_patient_timeline(patient_nbr)
    
    assert raw_record, "Could not fetch active patient timeline record"
    
    # Map Demographics & State
    demographics = raw_record.get("demographics", {})
    raw_timeline = raw_record.get("timeline", [])
    
    patient_state = PatientClinicalState(
        patient_nbr=patient_nbr,
        demographics=demographics,
        raw_timeline=raw_timeline
    )
    
    vector_db = LocalADAVectorStore()
    
    # Run the orchestrator in mock mode (which completes instantly and runs fully locally)
    orchestrator = PatientOrchestrator(patient_state, vector_db, use_mock_llm=True)
    report = orchestrator.run_orchestration()
    
    assert report is not None
    assert "DIA-TRACE.AI CLINICAL DECISION REPORT" in report
    assert "Causal Evidence Chain" in report
    assert "Recommendations" in report
    assert "Biomarker" in report or "Decline" in report
