import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sys
from typing import List, Dict, Any, Union

# Add root folder to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.database.db_handler import init_db, insert_patients_bulk, insert_encounters_bulk, insert_medications_bulk, insert_notes_bulk

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")
CSV_PATH = os.path.join(DATA_DIR, "diabetic_data.csv")
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

def download_and_extract_dataset():
    """Downloads the Diabetes 130-US Hospitals dataset ZIP from UCI and extracts it."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(CSV_PATH):
        print(f"Dataset already exists at: {CSV_PATH}. Skipping download.")
        return
        
    print(f"Downloading dataset from {UCI_ZIP_URL}...")
    try:
        import ssl
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(UCI_ZIP_URL, context=context) as response, open(ZIP_PATH, 'wb') as out_file:
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        print("Download complete. Extracting files...")
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
        # Check if the CSV was extracted inside a subfolder (usually 'dataset_diabetes')
        extracted_csv = os.path.join(DATA_DIR, "dataset_diabetes", "diabetic_data.csv")
        if os.path.exists(extracted_csv):
            os.rename(extracted_csv, CSV_PATH)
            # Clean up the subdirectory and other extracted files
            os.rename(os.path.join(DATA_DIR, "dataset_diabetes", "IDs_mapping.csv"), os.path.join(DATA_DIR, "IDs_mapping.csv"))
            os.rmdir(os.path.join(DATA_DIR, "dataset_diabetes"))
            
        # Remove zip file
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
            
        print(f"Dataset extracted and available at: {CSV_PATH}")
    except Exception as e:
        print(f"Error downloading or extracting dataset: {e}")
        # If the direct UCI link fails, raise so we know
        raise e

def clean_and_generate_cohort() -> pd.DataFrame:
    """
    Cleans raw CSV and filters for complex patient cohort (3+ encounters).
    Returns a sorted DataFrame.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cleaned CSV not found at: {CSV_PATH}")
        
    print("Loading dataset into pandas...")
    df = pd.read_csv(CSV_PATH)
    
    # Replace "?" placeholders with standard NaN/None
    df = df.replace("?", np.nan)
    
    # Filter out patients who have fewer than 3 encounters
    print("Filtering for longitudinal cohort (patients with >= 3 encounters)...")
    patient_counts = df["patient_nbr"].value_counts()
    multi_visit_patients = patient_counts[patient_counts >= 3].index
    
    cohort_df = df[df["patient_nbr"].isin(multi_visit_patients)].copy()
    
    # Sort chronologically by patient and encounter ID
    cohort_df = cohort_df.sort_values(by=["patient_nbr", "encounter_id"]).reset_index(drop=True)
    print(f"Cohort isolated: {len(multi_visit_patients)} patients with total of {len(cohort_df)} encounters.")
    
    return cohort_df

def is_kidney_related(code: str) -> bool:
    """Helper to check if ICD-9 code represents kidney complications/disease."""
    if pd.isna(code):
        return False
    code_str = str(code).strip()
    # Diabetic Nephropathy: 250.4
    # Chronic kidney disease: 580-589
    if code_str.startswith("250.4"):
        return True
    try:
        # Check numeric ranges
        val = float(code_str)
        if 580 <= val <= 589:
            return True
    except ValueError:
        pass
    return False

def is_nerve_related(code: str) -> bool:
    """Helper to check if ICD-9 code represents diabetic neuropathy."""
    if pd.isna(code):
        return False
    code_str = str(code).strip()
    # Diabetic Neuropathy: 250.6 or 357.2
    if code_str.startswith("250.6") or code_str.startswith("357.2"):
        return True
    return False

def is_eye_related(code: str) -> bool:
    """Helper to check if ICD-9 code represents diabetic retinopathy."""
    if pd.isna(code):
        return False
    code_str = str(code).strip()
    # Diabetic Retinopathy: 250.5 or 362.0
    if code_str.startswith("250.5") or code_str.startswith("362.0"):
        return True
    return False

def synthesize_eGFR_trajectory(group: pd.DataFrame) -> List[int]:
    """
    Computes a realistic chronological sequence of eGFR scores for a patient's visits.
    """
    n_visits = len(group)
    baseline = random.randint(70, 95)
    
    # Determine if they have any kidney complication in their diagnoses
    has_kidney_disease = False
    for code in list(group["diag_1"]) + list(group["diag_2"]) + list(group["diag_3"]):
        if is_kidney_related(code):
            has_kidney_disease = True
            break
            
    eGFRs = []
    current_eGFR = baseline
    for i in range(n_visits):
        if has_kidney_disease:
            # Drop steeply: 4 to 8 units per visit
            drop = random.randint(4, 8)
            current_eGFR -= drop
        else:
            # Drop slightly or fluctuate: -2 to +1
            current_eGFR += random.randint(-2, 1)
            
        # eGFR minimum is 10 (end-stage)
        current_eGFR = max(10, current_eGFR)
        eGFRs.append(current_eGFR)
        
    return eGFRs

def generate_clinician_note(row: pd.Series, visit_idx: int, total_visits: int, eGFR: int) -> str:
    """
    Programmatically synthesizes a highly realistic progress note based on structured patient metrics,
    using medical jargon, abbreviations, and symptom details.
    """
    age_str = str(row["age"]).replace("[", "").replace(")", "").replace("-", " to ")
    gender = "male" if row["gender"] == "Male" else "female"
    race = row["race"] if pd.notna(row["race"]) else "unknown race"
    
    # Extract medication status (24 standard drugs)
    drug_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", 
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", 
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", 
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    
    active_drugs = []
    adjustments = []
    for drug in drug_cols:
        if drug in row and pd.notna(row[drug]) and row[drug] != "No":
            status = row[drug]
            active_drugs.append(f"{drug.capitalize()} ({status})")
            if status in ["Up", "Down"]:
                adjustments.append(f"{drug.capitalize()} dosage adjusted {status.lower()}")
                
    active_drugs_str = ", ".join(active_drugs) if active_drugs else "No active oral or insulin therapies"
    
    # Biomarkers
    a1c = row["A1Cresult"]
    glu = row["max_glu_serum"]
    
    # Build clinical text paragraphs
    header = f"CLINICAL PROGRESS NOTE - VISIT {visit_idx} OF {total_visits}\n"
    header += f"Date: {datetime.now().strftime('%Y-%m-%d')} | Patient ID: {row['patient_nbr']} | Encounter: {row['encounter_id']}\n"
    header += f"Demographics: {age_str} y.o. {race} {gender}.\n"
    header += "---------------------------------------------------------\n"
    
    subjective = "SUBJECTIVE:\n"
    # General status
    subjective += f"Patient admitted for comprehensive metabolic review (stay duration: {row['time_in_hospital']} days). "
    
    # Medication compliance simulation
    gi_upset = False
    if "Metformin" in active_drugs_str and random.random() < 0.15:
        subjective += "Pt reports intermittent Metformin omission, citing transient GI upsets and nausea. "
        gi_upset = True
    else:
        subjective += "Patient reports compliance with current medications except during transient illnesses. "
        
    # Complication subjective reports
    has_neuro = is_nerve_related(row["diag_1"]) or is_nerve_related(row["diag_2"]) or is_nerve_related(row["diag_3"])
    has_nephro = is_kidney_related(row["diag_1"]) or is_kidney_related(row["diag_2"]) or is_kidney_related(row["diag_3"])
    has_retino = is_eye_related(row["diag_1"]) or is_eye_related(row["diag_2"]) or is_eye_related(row["diag_3"])
    
    if has_neuro:
        subjective += "Pt reports mild bilateral tingling and numbness in toes and lower extremities, worse at night. "
    if has_retino:
        subjective += "Complains of mild blurred vision and difficulty reading small print. "
    if not (has_neuro or has_retino):
        subjective += "Denies acute microvascular symptoms, tingling, or visual changes today. "
        
    objective = "\n\nOBJECTIVE:\n"
    objective += f"Physical Exam: Lower extremity sensation intact except for decreased monofilament response at bilateral toes. " if has_neuro else "Physical Exam: Lower extremity sensation normal, monofilament check intact. "
    objective += f"Telemetry: {row['num_lab_procedures']} lab panels reviewed. "
    
    # Biomarkers
    if pd.notna(a1c) and a1c != "None":
        objective += f"HbA1c levels recorded at {a1c}. "
        if a1c in [">8", ">7"]:
            objective += f"Glycemic threshold continues to remain elevated. "
    else:
        objective += "HbA1c was not checked during this window. "
        
    if pd.notna(glu) and glu != "None":
        objective += f"Random blood glucose spikes of {glu} noted. "
        
    objective += f"Calculated renal clearance (eGFR) is {eGFR} mL/min/1.73m2. "
    if eGFR < 60:
        objective += "Signs consistent with Moderate CKD (Stage 3). "
    elif eGFR < 30:
        objective += "Signs consistent with Severe CKD (Stage 4). "
        
    assessment = "\n\nASSESSMENT & PLAN:\n"
    assessment += f"1. Type 2 Diabetes Mellitus - "
    if pd.notna(a1c) and a1c in [">8", ">7"]:
        assessment += "poorly controlled, glycemic failure noted. "
    else:
        assessment += "suboptimally controlled. "
        
    # Complications assessment
    if has_nephro or eGFR < 60:
        assessment += "Complicated by diabetic nephropathy. "
    if has_neuro:
        assessment += "Complicated by peripheral neuropathy. "
    if has_retino:
        assessment += "Complicated by diabetic retinopathy. "
        
    # Dosing action
    assessment += f"\nActive Medication Matrix: {active_drugs_str}. "
    if adjustments:
        assessment += "Action taken: " + "; ".join(adjustments) + ". "
    else:
        assessment += "Current medication dosage maintained. "
        
    # GI side effects note
    if gi_upset:
        assessment += "Metformin side effects discussed. Plan to consider transition to SGLT2 inhibitor (e.g. Empagliflozin) or GLP-1 receptor agonist if glycemic failure and GI intolerance persist."
    elif eGFR < 45 and "Metformin" in active_drugs_str:
        assessment += "WARNING: eGFR approaching contraindication limit for Metformin. Close monitoring required; recommend dose reduction or class transition."
    elif eGFR < 30 and "Metformin" in active_drugs_str:
        assessment += "CRITICAL: eGFR < 30. Metformin contraindicated. Discontinue Metformin immediately. Recommend SGLT2 inhibitor therapy secondary line."
        
    return header + subjective + objective + assessment

def populate_database(cohort_df: pd.DataFrame):
    """
    Populates SQLite EHR database with cleaned patients, encounters, 
    medications, and synthetic progress notes.
    """
    print("Initializing database schema...")
    init_db()
    
    # 1. Dim Patients
    print("Preparing patient dimension records...")
    patient_cols = ["patient_nbr", "race", "gender", "age"]
    # Drop duplicates to isolate unique patients
    unique_patients = cohort_df[patient_cols].drop_duplicates(subset=["patient_nbr"]).to_dict("records")
    
    # Clean ages (ranges) in dictionaries
    for p in unique_patients:
        if pd.isna(p["race"]): p["race"] = "Unknown"
        if pd.isna(p["gender"]): p["gender"] = "Unknown"
        if pd.isna(p["age"]): p["age"] = "Unknown"
        
    print(f"Bulk loading {len(unique_patients)} patient profiles into dim_patients...")
    insert_patients_bulk(unique_patients)
    
    # 2. Map and generate eGFR and Notes chronologically
    print("Generating chronological eGFR profiles and clinician progress notes...")
    
    encounters_list = []
    medications_list = []
    notes_list = []
    
    # Group by patient to calculate sequential eGFR trajectories
    grouped = cohort_df.groupby("patient_nbr")
    
    patient_count = 0
    total_patients = len(grouped)
    
    drug_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", 
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", 
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", 
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    
    for patient_nbr, group in grouped:
        patient_count += 1
        if patient_count % 500 == 0 or patient_count == total_patients:
            print(f"Processing patient {patient_count}/{total_patients}...")
            
        eGFR_trajectory = synthesize_eGFR_trajectory(group)
        
        for idx, (_, row) in enumerate(group.iterrows()):
            visit_idx = idx + 1
            eGFR = eGFR_trajectory[idx]
            encounter_id = int(row["encounter_id"])
            
            # Prepare encounter fact record
            encounter_rec = {
                "encounter_id": encounter_id,
                "patient_nbr": int(patient_nbr),
                "time_in_hospital": int(row["time_in_hospital"]),
                "num_lab_procedures": int(row["num_lab_procedures"]),
                "num_procedures": int(row["num_procedures"]),
                "num_medications": int(row["num_medications"]),
                "number_outpatient": int(row["number_outpatient"]),
                "number_emergency": int(row["number_emergency"]),
                "number_inpatient": int(row["number_inpatient"]),
                "diag_1": row["diag_1"] if pd.notna(row["diag_1"]) else None,
                "diag_2": row["diag_2"] if pd.notna(row["diag_2"]) else None,
                "diag_3": row["diag_3"] if pd.notna(row["diag_3"]) else None,
                "number_diagnoses": int(row["number_diagnoses"]),
                "max_glu_serum": row["max_glu_serum"] if pd.notna(row["max_glu_serum"]) else None,
                "A1Cresult": row["A1Cresult"] if pd.notna(row["A1Cresult"]) else None,
                "eGFR": eGFR,
                "readmitted": row["readmitted"] if pd.notna(row["readmitted"]) else None
            }
            encounters_list.append(encounter_rec)
            
            # Prepare active medications
            for drug in drug_cols:
                if drug in row and pd.notna(row[drug]) and row[drug] != "No":
                    medications_list.append({
                        "encounter_id": encounter_id,
                        "patient_nbr": int(patient_nbr),
                        "drug_name": drug,
                        "dosage_status": row[drug]
                    })
                    
            # Synthesize notes
            note_text = generate_clinician_note(row, visit_idx, len(group), eGFR)
            # Create sequential visit date (spacing each encounter by ~6 months)
            visit_date = (datetime.now() - timedelta(days=180 * (len(group) - visit_idx))).strftime("%Y-%m-%d")
            
            notes_list.append({
                "encounter_id": encounter_id,
                "patient_nbr": int(patient_nbr),
                "note_text": note_text,
                "created_date": visit_date
            })
            
    print(f"Bulk loading {len(encounters_list)} encounter records into fact_encounters...")
    insert_encounters_bulk(encounters_list)
    
    print(f"Bulk loading {len(medications_list)} medication status records into fact_medications...")
    insert_medications_bulk(medications_list)
    
    print(f"Bulk loading {len(notes_list)} synthetic progress notes into synthetic_notes...")
    insert_notes_bulk(notes_list)
    
    print("Database loading completed successfully!")

if __name__ == "__main__":
    print("--- Starting DiaTrace.AI EHR Ingestion Pipeline ---")
    try:
        download_and_extract_dataset()
        cohort_df = clean_and_generate_cohort()
        populate_database(cohort_df)
        print("--- Ingestion Pipeline Finished Successfully ---")
    except Exception as e:
        print(f"FATAL ERROR in ingestion pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
