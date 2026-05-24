import urllib.request
import json
import sys

print("=== Starting End-to-End API Integration Test ===")

try:
    # 1. Fetch patients list from SQLite database
    print("\n[Step 1] Fetching patient records list from API...")
    with urllib.request.urlopen("http://localhost:8000/api/patients", timeout=5) as response:
        patients = json.loads(response.read().decode())
        print(f"-> Success! Mapped {len(patients)} patient records from database.")
        
        if not patients:
            print("Error: No patients found in database!")
            sys.exit(1)
            
        target_patient = patients[0]['patient_nbr']
        print(f"-> Selected Patient ID: {target_patient} (Visit count: {patients[0]['visit_count']})")
        
    # 2. Trigger live multi-agent analysis
    print(f"\n[Step 2] Launching live multi-agent clinical analysis for Patient #{target_patient}...")
    print("-> Note: This pings port 11434, detects Ollama, and executes the 4 sequential CrewAI agents. Please wait...")
    
    req = urllib.request.Request(
        "http://localhost:8000/api/analyze",
        data=json.dumps({"patient_nbr": target_patient}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with urllib.request.urlopen(req, timeout=120) as analysis_res:
        res_data = json.loads(analysis_res.read().decode())
        
        print("\n=== E2E PIPELINE RUN COMPLETED SUCCESSFULLY! ===")
        print(f"Patient Record:  {res_data['patient_nbr']}")
        print(f"Demographics:    {res_data['demographics']}")
        print(f"eGFR Slope:      {res_data['biomarkers']['egfr_slope']} mL/min/year")
        print(f"Complications:   {res_data['complications']}")
        print(f"Guideline Clashes: {res_data['ada_clashes']}")
        print("\nActionable Recommendations:")
        for idx, rec in enumerate(res_data['recommendations'], 1):
            print(f" {idx}. {rec}")
            
except Exception as e:
    print(f"\n❌ E2E Integration test failed: {e}")
    sys.exit(1)
