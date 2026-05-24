import json
from typing import Any

# Try to import crewai components
try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False

def create_temporal_miner_agent(llm: Any) -> Any:
    """Creates the Senior Clinical Informationist (Temporal Miner) Agent."""
    if not HAS_CREWAI:
        return None
        
    return Agent(
        role="Senior Clinical Informationist (Temporal Miner)",
        goal="Ingest patient notes chronologically and parse them into a structured medication, symptom, and biomarker timeline.",
        backstory="An expert in medical informatics and clinical data wrangling. You extract key longitudinal clinical indicators from raw physician text.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def create_temporal_miner_task(agent: Any, patient_nbr: int, raw_timeline: list) -> Any:
    """Creates the CrewAI task for the Temporal Miner Agent."""
    if not HAS_CREWAI:
        return None
        
    return Task(
        description=f"Parse the following raw timeline encounters for patient ID {patient_nbr}:\n"
                    f"{json.dumps(raw_timeline, indent=2)}\n"
                    f"Output a clean chronological mapping of medications, laboratory biomarkers, and symptoms.",
        expected_output="A structured list of chronological visit indexes with extracted biomarkers (eGFR, A1c) and medications.",
        agent=agent
    )
