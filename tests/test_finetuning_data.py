import os
import json
import pytest

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "instruction_dataset.jsonl"))

def test_instruction_dataset_exists():
    """Verify that instruction_dataset.jsonl was successfully generated."""
    assert os.path.exists(DATASET_PATH), f"Dataset not found at {DATASET_PATH}. Run generation pipeline first."

def test_dataset_records_structure():
    """Verify the formatting structure of instruction-tuning records (Alpaca style)."""
    if not os.path.exists(DATASET_PATH):
        pytest.skip("Instruction dataset is missing. Skipping test.")
        
    records_checked = 0
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 500:  # Check a representative sample of 500 records for performance
                break
                
            record = json.loads(line)
            
            # Assert core instruction-tuning keys
            assert "instruction" in record, f"Record {i} missing 'instruction' key"
            assert "input" in record, f"Record {i} missing 'input' key"
            assert "output" in record, f"Record {i} missing 'output' key"
            
            # Verify system instruction content
            assert "medical entity extraction" in record["instruction"].lower()
            
            # Verify input note structure
            note_text = record["input"]
            assert "CLINICAL PROGRESS NOTE" in note_text
            assert "SUBJECTIVE:" in note_text
            assert "OBJECTIVE:" in note_text
            assert "ASSESSMENT & PLAN:" in note_text
            
            # Verify output is valid serialized JSON
            try:
                output_data = json.loads(record["output"])
            except json.JSONDecodeError:
                pytest.fail(f"Record {i} output is not valid JSON: {record['output']}")
                
            # Assert structured entity schema compliance
            assert "biomarkers" in output_data, f"Record {i} output missing 'biomarkers' key"
            assert "medications" in output_data, f"Record {i} output missing 'medications' key"
            
            biomarkers = output_data["biomarkers"]
            assert "eGFR" in biomarkers, f"Record {i} biomarkers missing 'eGFR'"
            assert "HbA1c" in biomarkers, f"Record {i} biomarkers missing 'HbA1c'"
            assert "Glucose" in biomarkers, f"Record {i} biomarkers missing 'Glucose'"
            
            # eGFR must be numeric (int/float) or "Not Tested"
            egfr = biomarkers["eGFR"]
            assert isinstance(egfr, (int, float)) or egfr == "Not Tested"
            
            # Medications list structure
            meds = output_data["medications"]
            assert isinstance(meds, list), f"Record {i} medications is not a list"
            for m in meds:
                assert "name" in m, f"Medication in record {i} missing 'name'"
                assert "status" in m, f"Medication in record {i} missing 'status'"
                assert m["status"] in ["Steady", "Up", "Down", "No"], f"Medication status '{m['status']}' invalid in record {i}"
                
            records_checked += 1
            
    assert records_checked > 0, "No records were validated"
    print(f"\nSuccessfully validated {records_checked} instruction-tuning records against the strict medical schema.")
