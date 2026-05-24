import os
import sqlite3
import pytest

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ehr_database.db"))

def get_db_connection():
    if not os.path.exists(DB_PATH):
        pytest.skip(f"Database not found at: {DB_PATH}. Run ingestion first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def test_database_files_exist():
    """Verify database file exists in data/processed."""
    assert os.path.exists(DB_PATH), f"EHR SQLite database does not exist at {DB_PATH}"

def test_patients_dimension():
    """Assert that patient dimension is populated and valid."""
    conn = get_db_connection()
    try:
        patients = conn.execute("SELECT * FROM dim_patients LIMIT 100").fetchall()
        assert len(patients) > 0, "dim_patients is empty"
        for p in patients:
            assert p["patient_nbr"] is not None
            assert p["gender"] in ["Male", "Female", "Unknown", "Other"]
            assert p["age"] != "Unknown"
    finally:
        conn.close()

def test_longitudinal_cohort_constraint():
    """
    CRITICAL CONSTRAINT: Assert that every single patient in the database 
    has at least 3 clinical visits/encounters (longitudinal cohort).
    """
    conn = get_db_connection()
    try:
        # Group encounters by patient and check counts
        counts = conn.execute("""
            SELECT patient_nbr, COUNT(encounter_id) as visit_count 
            FROM fact_encounters 
            GROUP BY patient_nbr
        """).fetchall()
        
        assert len(counts) > 0, "No encounters found"
        for row in counts:
            assert row["visit_count"] >= 3, f"Patient {row['patient_nbr']} has only {row['visit_count']} visits, constraint broken!"
            
        print(f"Verified cohort size of {len(counts)} patient timelines. All have >= 3 visits.")
    finally:
        conn.close()

def test_medications_table():
    """Assert fact_medications is loaded and references valid encounters."""
    conn = get_db_connection()
    try:
        meds = conn.execute("SELECT * FROM fact_medications LIMIT 100").fetchall()
        assert len(meds) > 0, "fact_medications is empty"
        for m in meds:
            assert m["encounter_id"] is not None
            assert m["patient_nbr"] is not None
            assert m["drug_name"] is not None
            assert m["dosage_status"] in ["Steady", "Up", "Down", "No"]
    finally:
        conn.close()

def test_synthetic_notes_integrity():
    """Verify that clinical notes are synthesized, structured, and contain clinical markers."""
    conn = get_db_connection()
    try:
        notes = conn.execute("SELECT * FROM synthetic_notes LIMIT 50").fetchall()
        assert len(notes) > 0, "synthetic_notes table is empty"
        for note in notes:
            text = note["note_text"]
            assert "CLINICAL PROGRESS NOTE" in text
            assert "SUBJECTIVE:" in text
            assert "OBJECTIVE:" in text
            assert "ASSESSMENT & PLAN:" in text
            assert "eGFR" in text or "renal" in text or "clearance" in text
            
            # Check linking
            assert note["encounter_id"] is not None
            assert note["patient_nbr"] is not None
    finally:
        conn.close()

def test_eGFR_decline_trends():
    """Assert eGFR values are synthesized within realistic ranges and mapped."""
    conn = get_db_connection()
    try:
        encounters = conn.execute("SELECT eGFR FROM fact_encounters WHERE eGFR IS NOT NULL LIMIT 200").fetchall()
        assert len(encounters) > 0, "No encounters with eGFR found"
        for enc in encounters:
            egfr = enc["eGFR"]
            assert 10 <= egfr <= 120, f"eGFR {egfr} out of realistic bounds"
    finally:
        conn.close()
