import os
import sqlite3
from typing import Dict, List, Any, Optional

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
DB_PATH = os.path.join(DB_DIR, "ehr_database.db")

def get_connection() -> sqlite3.Connection:
    """Creates directory if not exists and returns a connection to SQLite."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the Star Schema SQLite tables for DiaTrace.AI EHR."""
    conn = get_connection()
    try:
        with conn:
            # 1. Dimension Patients Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dim_patients (
                    patient_nbr INTEGER PRIMARY KEY,
                    race TEXT,
                    gender TEXT,
                    age TEXT
                );
            """)

            # 2. Fact Encounters Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fact_encounters (
                    encounter_id INTEGER PRIMARY KEY,
                    patient_nbr INTEGER,
                    time_in_hospital INTEGER,
                    num_lab_procedures INTEGER,
                    num_procedures INTEGER,
                    num_medications INTEGER,
                    number_outpatient INTEGER,
                    number_emergency INTEGER,
                    number_inpatient INTEGER,
                    diag_1 TEXT,
                    diag_2 TEXT,
                    diag_3 TEXT,
                    number_diagnoses INTEGER,
                    max_glu_serum TEXT,
                    A1Cresult TEXT,
                    eGFR INTEGER,
                    readmitted TEXT,
                    FOREIGN KEY(patient_nbr) REFERENCES dim_patients(patient_nbr)
                );
            """)

            # 3. Fact Medications Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fact_medications (
                    medication_instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    encounter_id INTEGER,
                    patient_nbr INTEGER,
                    drug_name TEXT,
                    dosage_status TEXT,
                    FOREIGN KEY(encounter_id) REFERENCES fact_encounters(encounter_id),
                    FOREIGN KEY(patient_nbr) REFERENCES dim_patients(patient_nbr)
                );
            """)

            # 4. Synthetic Notes Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS synthetic_notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    encounter_id INTEGER,
                    patient_nbr INTEGER,
                    note_text TEXT,
                    created_date TEXT,
                    FOREIGN KEY(encounter_id) REFERENCES fact_encounters(encounter_id),
                    FOREIGN KEY(patient_nbr) REFERENCES dim_patients(patient_nbr)
                );
            """)
    finally:
        conn.close()

def insert_patients_bulk(patients: List[Dict[str, Any]]):
    """Inserts multiple patients in a single transaction."""
    conn = get_connection()
    try:
        with conn:
            conn.executemany("""
                INSERT OR IGNORE INTO dim_patients (patient_nbr, race, gender, age)
                VALUES (:patient_nbr, :race, :gender, :age)
            """, patients)
    finally:
        conn.close()

def insert_encounters_bulk(encounters: List[Dict[str, Any]]):
    """Inserts multiple encounters in a single transaction."""
    conn = get_connection()
    try:
        with conn:
            conn.executemany("""
                INSERT OR IGNORE INTO fact_encounters (
                    encounter_id, patient_nbr, time_in_hospital, num_lab_procedures,
                    num_procedures, num_medications, number_outpatient, number_emergency,
                    number_inpatient, diag_1, diag_2, diag_3, number_diagnoses,
                    max_glu_serum, A1Cresult, eGFR, readmitted
                ) VALUES (
                    :encounter_id, :patient_nbr, :time_in_hospital, :num_lab_procedures,
                    :num_procedures, :num_medications, :number_outpatient, :number_emergency,
                    :number_inpatient, :diag_1, :diag_2, :diag_3, :number_diagnoses,
                    :max_glu_serum, :A1Cresult, :eGFR, :readmitted
                )
            """, encounters)
    finally:
        conn.close()

def insert_medications_bulk(medications: List[Dict[str, Any]]):
    """Inserts multiple medication entries in a single transaction."""
    conn = get_connection()
    try:
        with conn:
            conn.executemany("""
                INSERT INTO fact_medications (encounter_id, patient_nbr, drug_name, dosage_status)
                VALUES (:encounter_id, :patient_nbr, :drug_name, :dosage_status)
            """, medications)
    finally:
        conn.close()

def insert_notes_bulk(notes: List[Dict[str, Any]]):
    """Inserts multiple synthetic clinician progress notes in a single transaction."""
    conn = get_connection()
    try:
        with conn:
            conn.executemany("""
                INSERT INTO synthetic_notes (encounter_id, patient_nbr, note_text, created_date)
                VALUES (:encounter_id, :patient_nbr, :note_text, :created_date)
            """, notes)
    finally:
        conn.close()

def get_patient_timeline(patient_nbr: int) -> Dict[str, Any]:
    """
    Queries and returns a patient's demographics, clinical encounters, 
    medication changes, and clinical text notes over time.
    """
    conn = get_connection()
    try:
        # Demographics
        patient_row = conn.execute(
            "SELECT * FROM dim_patients WHERE patient_nbr = ?", (patient_nbr,)
        ).fetchone()
        
        if not patient_row:
            return {}

        patient_info = dict(patient_row)

        # Encounters ordered by encounter_id chronologically
        encounter_rows = conn.execute(
            "SELECT * FROM fact_encounters WHERE patient_nbr = ? ORDER BY encounter_id ASC", 
            (patient_nbr,)
        ).fetchall()
        
        encounters = [dict(row) for row in encounter_rows]
        encounter_ids = [enc["encounter_id"] for enc in encounters]

        # Medications for these encounters
        medications = []
        if encounter_ids:
            placeholders = ",".join("?" for _ in encounter_ids)
            med_rows = conn.execute(
                f"SELECT * FROM fact_medications WHERE encounter_id IN ({placeholders})", 
                encounter_ids
            ).fetchall()
            medications = [dict(row) for row in med_rows]

        # Notes for these encounters
        notes = []
        if encounter_ids:
            placeholders = ",".join("?" for _ in encounter_ids)
            note_rows = conn.execute(
                f"SELECT * FROM synthetic_notes WHERE encounter_id IN ({placeholders}) ORDER BY encounter_id ASC", 
                encounter_ids
            ).fetchall()
            notes = [dict(row) for row in note_rows]

        # Group medications by encounter
        meds_by_encounter = {}
        for med in medications:
            enc_id = med["encounter_id"]
            if enc_id not in meds_by_encounter:
                meds_by_encounter[enc_id] = []
            meds_by_encounter[enc_id].append({
                "drug_name": med["drug_name"],
                "dosage_status": med["dosage_status"]
            })

        # Compile chronological history
        timeline = []
        for i, enc in enumerate(encounters):
            enc_id = enc["encounter_id"]
            matching_note = next((n["note_text"] for n in notes if n["encounter_id"] == enc_id), None)
            timeline.append({
                "visit_index": i + 1,
                "encounter_id": enc_id,
                "time_in_hospital": enc["time_in_hospital"],
                "num_lab_procedures": enc["num_lab_procedures"],
                "num_procedures": enc["num_procedures"],
                "num_medications": enc["num_medications"],
                "diagnoses": [enc["diag_1"], enc["diag_2"], enc["diag_3"]],
                "A1Cresult": enc["A1Cresult"],
                "max_glu_serum": enc["max_glu_serum"],
                "eGFR": enc["eGFR"],
                "medication_adjustments": meds_by_encounter.get(enc_id, []),
                "clinician_note": matching_note
            })

        return {
            "patient_nbr": patient_nbr,
            "demographics": patient_info,
            "timeline": timeline
        }
    finally:
        conn.close()
