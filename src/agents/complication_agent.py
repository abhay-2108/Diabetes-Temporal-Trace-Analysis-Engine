from typing import Any

# Try to import crewai components
try:
    from crewai import Agent, Task
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False

def create_complication_agent(llm: Any, tools: list) -> Any:
    """Creates the Expert Clinical Pathologist (Complication Tracker) Agent."""
    if not HAS_CREWAI:
        return None
        
    return Agent(
        role="Expert Clinical Pathologist (Complication Tracker)",
        goal="Analyze structured timelines and biomarkers mathematically to identify hidden microvascular kidney, nerve, or eye complications.",
        backstory="A clinical epidemiologist specializing in chronic disease progression. You calculate rates of decline and map sub-clinical target organ damage.",
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm
    )

def create_complication_task(agent: Any, patient_nbr: int) -> Any:
    """Creates the CrewAI task for the Complication Agent."""
    if not HAS_CREWAI:
        return None
        
    return Task(
        description=f"Analyze the parsed patient timeline. Use the Mathematical Complication Slope Analyzer Tool on patient {patient_nbr} "
                    f"to calculate the exact renal clearance slope (eGFR rate of decline) and identify nerve or eye symptoms in clinical narratives.",
        expected_output="An expert clinical microvascular complication risk assessment with mathematical decline rates and clinical symptom mapping.",
        agent=agent
    )
