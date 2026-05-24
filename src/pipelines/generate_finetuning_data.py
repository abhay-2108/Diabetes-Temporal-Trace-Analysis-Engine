import os
import sqlite3
import json
import random

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "ehr_database.db"))
OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "instruction_dataset.jsonl"))

def generate_dataset():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Query to join encounters, notes, and medications
    query = """
    SELECT 
        e.encounter_id,
        n.note_text,
        e.eGFR,
        e.A1Cresult,
        e.max_glu_serum
    FROM synthetic_notes n
    JOIN fact_encounters e ON n.encounter_id = e.encounter_id
    """
    
    rows = conn.execute(query).fetchall()
    
    dataset = []
    
    system_instruction = "You are a medical entity extraction system. Given a clinical progress note, extract the patient's biomarkers and active medications into a strict JSON format."
    
    for row in rows:
        encounter_id = row["encounter_id"]
        note_text = row["note_text"]
        
        # Get medications for this encounter
        meds = conn.execute("SELECT drug_name, dosage_status FROM fact_medications WHERE encounter_id = ?", (encounter_id,)).fetchall()
        
        medication_list = [{"name": m["drug_name"], "status": m["dosage_status"]} for m in meds]
        
        # Construct the target JSON
        target_json = {
            "biomarkers": {
                "eGFR": row["eGFR"],
                "HbA1c": row["A1Cresult"] if row["A1Cresult"] else "Not Tested",
                "Glucose": row["max_glu_serum"] if row["max_glu_serum"] else "Not Tested"
            },
            "medications": medication_list
        }
        
        # Format for Unsloth / Hugging Face Alpaca format
        record = {
            "instruction": system_instruction,
            "input": note_text,
            "output": json.dumps(target_json)
        }
        dataset.append(record)
        
    conn.close()
    
    # Shuffle for better training
    random.seed(42)
    random.shuffle(dataset)
    
    # Write to JSONL
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
            
    print(f"Successfully generated {len(dataset)} instruction-tuning pairs.")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    generate_dataset()
